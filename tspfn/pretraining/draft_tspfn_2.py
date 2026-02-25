import os
import csv
import logging
from sklearn.model_selection import train_test_split
from typing import Any, Callable, Dict, Literal, Optional, Sequence, Tuple, cast

import hydra
from omegaconf import DictConfig
import torch
from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig
from torch.utils.data import DataLoader
from torch import Tensor, nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryF1Score,
    BinaryAveragePrecision,
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassAveragePrecision,
    MulticlassF1Score,
)
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError
from torchmetrics import MetricCollection

from data.utils.decorators import auto_move_data
from tspfn.system import TSPFNSystem

logger = logging.getLogger(__name__)

# TODO: Split in advance the sequence of each dataset and export them as chunks of 1024 samples to avoid doing it on the fly because of RAM issues.


class TSPFNPretraining(TSPFNSystem):
    """Multi-modal transformer to learn a representation from cardiac imaging and patient records data."""

    def __init__(
        self,
        embed_dim: int,
        split_finetuning: float = 0.5,
        predict_losses: Optional[Dict[str, Callable[[Tensor, Tensor], Tensor]] | DictConfig] = None,
        *args,
        **kwargs,
    ):
        """Initializes class instance.

        Args:
            embed_dim: Size of the tokens/embedding for all the modalities.
            predict_losses: Supervised criteria to measure the error between the predicted attributes and their real
                value.
            *args: Positional arguments to pass to the parent's constructor.
            **kwargs: Keyword arguments to pass to the parent's constructor.
        """
        # Ensure string tags are converted to their appropriate enum types
        # And do it before call to the parent's `init` so that the converted values are saved in `hparams`

        super().__init__(*args, **kwargs)

        # Add shortcut to lr to work with Lightning's learning rate finder
        self.hparams["lr"] = None

        # Configure losses/metrics to compute at each train/val/test step
        self.metrics = nn.ModuleDict()
        # if target in TabularAttribute.numerical_attrs():
        #     self.metrics[target] = MetricCollection([MeanAbsoluteError(), MeanSquaredError()])
        # elif target in TabularAttribute.binary_attrs():
        #     self.metrics[target] = MetricCollection(
        #         [
        #             BinaryAccuracy(),
        #             BinaryAUROC(),
        #             BinaryAveragePrecision(),
        #             BinaryF1Score(),
        #         ],
        #     )
        # else:  # attr in TabularAttribute.categorical_attrs()

        # Supervised losses and metrics
        self.predict_losses = {}
        if predict_losses:
            self.predict_losses = {
                target_task: (
                    hydra.utils.instantiate(target_loss) if isinstance(target_loss, DictConfig) else target_loss
                )
                for target_task, target_loss in predict_losses.items()
            }

        # Initialize transformer encoder and self-supervised + prediction heads
        self.encoder, self.prediction_heads = self.configure_model()

        # Initialize inference storage tensors
        self.ts_train_for_inference = torch.Tensor().to(self.device)
        self.y_train_for_inference = torch.Tensor().to(self.device)

    @property
    def example_input_array(self) -> Tensor:
        """Redefine example input array based on the cardiac attributes provided to the model."""
        # 2 is the size of the batch in the example
        num_classes = 10  # Default number of classes in TabPFN prediction head
        # labels = torch.cat([torch.randperm(5)]*2)
        labels = torch.arange(10) % num_classes
        labels = labels.repeat(16, 1)
        time_series_attrs = torch.randn(16, 10, 64)  # (B, S, T)
        ts_example_input = torch.cat([time_series_attrs, labels.unsqueeze(-1)], dim=2)  # (B, S, T+1)
        # num_classes = len(torch.unique(labels))
        return ts_example_input

    def configure_model(
        self,
    ) -> Tuple[nn.Module, nn.Module, Optional[nn.ModuleDict]]:
        """Build the model, which must return a transformer encoder, and self-supervised or prediction heads."""
        # Build the transformer encoder
        encoder = hydra.utils.instantiate(self.hparams["model"]["encoder"])

        # Build the prediction heads following the architecture proposed in
        # https://arxiv.org/pdf/2106.11959
        prediction_heads = None
        if self.predict_losses:
            prediction_heads = nn.ModuleDict()
            for target_task in self.predict_losses:
                prediction_heads[target_task] = hydra.utils.instantiate(
                    self.hparams["model"]["prediction_head"],
                )

        return encoder, prediction_heads

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        """Configure optimizer to ignore parameters that should remain frozen (e.g. tokenizers)."""
        # Frozen tabpfn encoder
        if self.hparams["model"].get("freeze_encoder", False):
            for param in self.encoder.parameters():
                param.requires_grad = False
        return super().configure_optimizers()

    @auto_move_data
    def process_data(
        self,
        time_series_attrs: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Tokenizes the input time-series attributes, providing a mask of non-missing attributes.

        Args:
            time_series_attrs: (B, S, T), Batch of time-series datasets, where the last feature for each sample is the label.

        Returns:
            - (B, Support, 1), Support set labels.
            - (B, Query, 1), Query set labels.
            - (B, S (=Support+Query), T), Time series input for .
        """

        # Tokenize the attributes
        assert time_series_attrs is not None, "At least time_series_attrs must be provided to process_data."

        ts = time_series_attrs[:, :, :-1]  # (B, S, T)
        # indices = torch.arange(ts.shape[0])
        indices = torch.arange(1024)  # Fix for pretraining with sequence length S =1024
        y = time_series_attrs[:, :, -1]  # (B, S, 1)

        if self.training or len(self.y_train_for_inference) == 0:
            if len(self.y_train_for_inference) == 0:
                indices = torch.arange(10)
            assert self.hparams["split_finetuning"] > 0.0, "split_finetuning must be > 0.0 when training."
            ts_support_list = []
            ts_query_list = []
            y_batch_support_list = []
            y_batch_query_list = []
            for dataset_idx, dataset_labels in enumerate(y):
                labels = dataset_labels.clone().cpu()
                try:
                    train_indices, test_indices = train_test_split(
                        indices,
                        test_size=self.hparams["split_finetuning"],
                        random_state=self.hparams["seed"],
                        stratify=labels,
                    )
                except ValueError:
                    if len(indices) == 1:
                        print("Sanity check with single sample for TS, skipping train/test split.")
                        train_indices, test_indices = indices, indices
                    else:
                        print("Stratified splitting failed, performing non-stratified split instead.")
                        train_indices, test_indices = train_test_split(
                            indices,
                            test_size=self.hparams["split_finetuning"],
                            random_state=self.hparams["seed"],
                        )

                ts_support = torch.as_tensor(ts[dataset_idx, train_indices, :], dtype=torch.float32)
                ts_query = torch.as_tensor(ts[dataset_idx, test_indices, :], dtype=torch.float32)
                ts_support_list.append(ts_support)
                ts_query_list.append(ts_query)
                y_batch_support_list.append(torch.as_tensor(y[dataset_idx, train_indices], dtype=torch.float32))
                y_batch_query_list.append(torch.as_tensor(y[dataset_idx, test_indices], dtype=torch.float32))

            ts = torch.cat([torch.stack(ts_support_list, dim=0), torch.stack(ts_query_list, dim=0)], dim=1).to(
                self.device
            )
            y_batch_support = torch.stack(y_batch_support_list, dim=0).to(self.device)
            y_batch_query = torch.stack(y_batch_query_list, dim=0).to(self.device)

        # if self.training or len(self.y_train_for_inference) == 0:
        #     y_batch_support = torch.as_tensor(y[train_indices], dtype=torch.float32).to(self.device)
        #     y_batch_query = torch.as_tensor(y[test_indices], dtype=torch.float32).to(self.device)
        else:
            y_batch_support = y.to(self.device)
            y_batch_query = y.to(self.device)
            ts = ts.to(self.device)

        if ts.ndim == 2:
            ts = ts.unsqueeze(1)

        return (
            y_batch_support,
            y_batch_query,
            ts,
        )

    @auto_move_data
    def encode(
        self,
        y_batch_support: Tensor,
        ts: Tensor,
    ) -> Tensor:
        """Embeds input sequences using the encoder model, optionally selecting/pooling output ts for the embedding.

        Args:
            y_batch_support: (S, 1), Support set labels.
            ts: (B, S (=Support+Query), T), Tokens to feed to the encoder.
        Returns: (B, Query, E), Embeddings of the input sequences.
        """

        if self.training or len(self.ts_train_for_inference) == 0:
            out_features = self.encoder(ts.transpose(0, 1), y_batch_support.transpose(0, 1))[:, :, -1, :]

        else:
            # Use train set as context for predicting the query set on val/test inference
            ts_full = torch.cat([self.ts_train_for_inference, ts], dim=0)
            y_train = self.y_train_for_inference
            out_features = self.encoder(ts_full.transpose(0, 1), y_train.transpose(0, 1))[:, :, -1, :]

        return out_features  # (B, Query, E)

    @auto_move_data
    def forward(
        self,
        time_series_attrs: Tensor,
        task: Literal["encode", "predict"] = "encode",
    ) -> Tensor | Dict[str, Tensor]:
        """Performs a forward pass through i) the tokenizer, ii) the transformer encoder and iii) the prediction head.

        Args:
            time_series_attrs: (B, S, T): time series inputs.
            task: Flag indicating which type of inference task to perform.

        Returns:
            if `task` == 'encode':
                (B, Query, E), Batch of features extracted by the encoder.
            if `task` == 'predict' (and the model includes prediction heads):
                1 * (B, Query), Prediction for each target in `losses`.
        """
        if task != "encode" and not self.prediction_heads:
            raise ValueError(
                "You requested to perform a prediction task, but the model does not include any prediction heads."
            )
        y_batch_support, y_batch_query, ts = self.process_data(
            time_series_attrs,
        )  # (B, Support, 1), (B, Query, 1), (B, S, T)

        out_features = self.encode(y_batch_support, ts)  # (B, S, E) -> (B, E)

        # Early return if requested task requires no prediction heads
        if task == "encode":
            return out_features

        elif task == "predict":
            assert (
                self.prediction_heads is not None
            ), "You requested to perform a prediction task, but the model does not include any prediction heads."

            # Forward pass through each target's prediction head
            predictions = {
                target_task: prediction_head(out_features)
                for target_task, prediction_head in self.prediction_heads.items()
            }

            # Squeeze out the singleton dimension from the predictions' features (only relevant for scalar predictions)
            predictions = {target_task: prediction.squeeze(dim=1) for target_task, prediction in predictions.items()}
            return predictions

        else:
            raise ValueError(f"Unknown task '{task}' requested for forward pass.")

    @auto_move_data
    def get_latent_vectors(
        self,
        batch: Tensor,
        batch_idx: int,
    ) -> Tensor:
        """Extracts the latent vectors from the encoder for the given batch."""
        time_series_input = batch  # (B, S, T)
        # num_classes = num_classes.cpu().item()

        y_batch_support, y_batch_query, ts = self.process_data(time_series_attrs=time_series_input)  # (B, S, E), (B, S)
        return self.encode(
            y_batch_support,
            ts,
        )  # (B, S, E) -> (B, E)

    def _shared_step(self, batch: Tensor, batch_idx: int) -> Dict[str, Tensor]:
        # Extract time-series inputs from the batch
        time_series_input = batch
        # num_classes = num_classes[0].cpu().item()
        y_batch_support, y_batch_query, ts = self.process_data(time_series_attrs=time_series_input)  # (B, S, E), (B, S)

        metrics = {}
        losses = []
        if self.predict_losses is not None:
            metrics.update(self._prediction_shared_step(y_batch_support, y_batch_query, ts, num_classes=10))
            losses.append(metrics["s_loss"])

        # Compute the sum of the (weighted) losses
        metrics["loss"] = sum(losses)

        return metrics

    def _prediction_shared_step(
        self,
        y_batch_support: Tensor,
        y_batch_query: Tensor,
        ts: Tensor,
        num_classes: int,
    ) -> Dict[str, Tensor]:
        # Forward pass through the encoder without gradient computation to fine-tune only the prediction heads
        assert (
            self.prediction_heads is not None
        ), "You requested to perform a prediction task, but the model does not include any prediction heads."
        prediction = self.encode(y_batch_support, ts)
        predictions = {}
        for target_task, prediction_head in self.prediction_heads.items():
            pred = prediction_head(prediction)
            predictions[target_task] = pred

        for target_task in self.predict_losses:
            self.metrics[target_task] = MetricCollection(
                [
                    MulticlassAccuracy(num_classes=num_classes, average="micro"),
                    MulticlassAUROC(num_classes=num_classes, average="macro"),
                    MulticlassAveragePrecision(num_classes=num_classes, average="macro"),
                    MulticlassF1Score(num_classes=num_classes, average="macro"),
                ]
            ).to(self.device)

        # Compute the loss/metrics for each target label, ignoring items for which targets are missing
        losses, metrics = {}, {}

        target_batch = y_batch_query

        for target_task, target_loss in self.predict_losses.items():
            target, y_hat = target_batch, predictions[target_task]

            target = target.long()

            losses[f"{target_loss.__class__.__name__.lower().replace('loss', '')}/{target_task}"] = target_loss(
                y_hat,
                target,
            )

            for metric_tag, metric in self.metrics[target_task].items():
                metric.update(y_hat, target)

        losses["s_loss"] = torch.stack(list(losses.values())).mean()
        metrics.update(losses)

        return metrics

    def _on_epoch_start(self, dataloader: DataLoader):
        if dataloader is None:
            return "No training dataloader found while setting up inference storage tensors"
        else:
            if callable(dataloader):
                dataloader = dataloader()

            ts_batch_list = [ts_batch for ts_batch in dataloader]
            ts_tokens_support = torch.vstack(ts_batch_list)

            if ts_tokens_support.shape[0] > self.hparams["max_batches_stored_for_inference"]:
                # Randomly subsample to limit RAM usage
                perm = torch.randperm(ts_tokens_support.shape[0])
                ts_tokens_support = ts_tokens_support[perm[: self.hparams["max_batches_stored_for_inference"]]]

            assert ts_tokens_support.ndim == 3, "Input time-series tokens must have 3 dimensions (B, S, T+1)."

            # Store and remove label
            self.y_train_for_inference = ts_tokens_support[:, :, -1].to(self.device)
            ts_tokens_support = ts_tokens_support[:, :, :-1].to(self.device)

            if ts_tokens_support.ndim == 2:
                ts_tokens_support = ts_tokens_support.unsqueeze(1)

            self.ts_train_for_inference = ts_tokens_support

    def on_validation_epoch_start(self):
        self._on_epoch_start(self.trainer.val_dataloaders)

    def on_test_epoch_start(self):
        out_message = self._on_epoch_start(self.trainer.test_dataloaders)
        if out_message is not None:
            logger.info(f"Test epoch start: {out_message}")
            raise ValueError(out_message)

    def on_test_epoch_end(self):
        all_metrics = {}
        for target_task in self.predict_losses:
            for metric_tag, metric in self.metrics[target_task].items():
                metrics_value = metric.compute()
                self.log(f"test_{metric_tag}/{target_task}", metrics_value)
                all_metrics[f"{metric_tag}/{target_task}"] = (
                    metrics_value.item() if hasattr(metrics_value, "item") else metrics_value
                )
                metric.reset()
        output_dir = os.getcwd()
        csv_file = "test_metrics.csv"
        with open(csv_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            # Write headers
            writer.writerow(["metric", "value"])
            # Write metric data
            for key, value in all_metrics.items():
                writer.writerow([key, value])

        # Print metrics to terminal
        logger.info(f"Test metrics: {all_metrics}")

        # Reset inference storage tensors for next dataset
        # self.y_train_for_inference = torch.Tensor().to(self.device)
        # self.ts_train_for_inference = torch.Tensor().to(self.device)
