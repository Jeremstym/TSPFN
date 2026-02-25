import os
import csv
import logging
import numpy as np
import pandas as pd
from typing import Any, Callable, Dict, Literal, Optional, Sequence, Tuple, cast, Union

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
    BinaryCohenKappa,
    BinaryRecall,
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassF1Score,
    MulticlassAveragePrecision,
    MulticlassCohenKappa,
    MulticlassRecall,
)
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError
from torchmetrics import MetricCollection

from data.utils.decorators import auto_move_data
from tspfn.system import TSPFNSystem
from tspfn.foundationals.labram import TimeSeriesLabramEncoder
from tspfn.foundationals.convolution import TimeSeriesConvolutionTokenizer
from tspfn.utils import get_sizes_per_class, MulticlassFaiss, SingleclassFaiss, stratified_batch_split, half_batch_split

logger = logging.getLogger(__name__)


class TSPFNFineTuning(TSPFNSystem):
    """Multi-modal transformer to learn a representation from cardiac imaging and patient records data."""

    def __init__(
        self,
        embed_dim: int,
        split_finetuning: float = 0.5,
        predict_losses: Optional[Dict[str, Callable[[Tensor, Tensor], Tensor]] | DictConfig] = None,
        time_series_positional_encoding: Literal["none", "sinusoidal", "learned"] = "none",
        time_series_num_channels: int = 16,
        time_series_length: int = 1000,
        foundation_model_name: Literal["convolution", "labram"] = "labram",
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

        self.ts_num_channels = time_series_num_channels
        self.ts_length = time_series_length

        self.time_series_positional_encoding = time_series_positional_encoding
        self.time_series_convolution = nn.Sequential(
            nn.Conv1d(
                in_channels=self.ts_num_channels,
                out_channels=self.ts_num_channels,
                kernel_size=10,
                stride=10,
                groups=self.ts_num_channels,
            ),  # T = 1000 -> 100
            nn.ReLU(),
            nn.Conv1d(
                in_channels=self.ts_num_channels,
                out_channels=self.ts_num_channels,
                kernel_size=5,
                stride=5,
                groups=self.ts_num_channels,
            ),  # T = 100 -> 20
        )
        # self.time_series_convolution = nn.Sequential(
        #     nn.Conv1d(in_channels=self.ts_num_channels, out_channels=self.ts_num_channels, kernel_size=8, stride=4, padding=2, groups=self.ts_num_channels),
        #     nn.ReLU(),
        #     nn.Conv1d(in_channels=self.ts_num_channels, out_channels=self.ts_num_channels, kernel_size=5, stride=2, padding=2, groups=self.ts_num_channels),
        #     nn.ReLU(),
        #     nn.Conv1d(in_channels=self.ts_num_channels, out_channels=self.ts_num_channels, kernel_size=5, stride=2, padding=1, groups=self.ts_num_channels),
        #     nn.AdaptiveAvgPool1d(60),
        # )
        # self.time_series_convolution = nn.Sequential(
        #     nn.Conv1d(in_channels=self.ts_num_channels, out_channels=self.ts_num_channels, kernel_size=20, stride=2, groups=self.ts_num_channels), # T = 1000 -> 491
        #     nn.GELU(),
        #     nn.Conv1d(in_channels=self.ts_num_channels, out_channels=self.ts_num_channels, kernel_size=20, stride=5, groups=self.ts_num_channels), # T = 491 -> 95
        # )

        # Use ModuleDict so metrics move to GPU automatically
        metrics_template = MetricCollection(
            [
                MulticlassAccuracy(num_classes=6, average="micro"),
                MulticlassAUROC(num_classes=6, average="macro"),
                MulticlassAveragePrecision(num_classes=6, average="macro"),
                MulticlassF1Score(num_classes=6, average="weighted"),
                MulticlassCohenKappa(num_classes=6),
                MulticlassRecall(num_classes=6, average="macro"),
            ]
        )
        # Store them in a dict of ModuleDicts
        self.metrics = nn.ModuleDict(
            {
                "train_metrics": nn.ModuleDict({t: metrics_template.clone(prefix="train/") for t in predict_losses}),
                "val_metrics": nn.ModuleDict({t: metrics_template.clone(prefix="val/") for t in predict_losses}),
                "test_metrics": nn.ModuleDict({t: metrics_template.clone(prefix="test/") for t in predict_losses}),
            }
        )
        if foundation_model_name == "convolution":
            self.ts_tokenizer = TimeSeriesConvolutionTokenizer(
                ts_size=time_series_length,
                ts_num_channels=time_series_num_channels,
            )
        elif foundation_model_name == "labram":
            self.ts_tokenizer = TimeSeriesLabramEncoder(
                pretrained_weights="/home/stympopper/pretrainingTSPFN/ckpts/labram-base.pth",
            )
        elif foundation_model_name is None:
            self.ts_tokenizer = None
        else:
            raise ValueError(f"Unknown foundation model name '{foundation_model_name}' provided.")

    @property
    def example_input_array(self) -> Tensor:
        """Redefine example input array based on the cardiac attributes provided to the model."""
        batch_size = 10
        num_classes = 6  # Default number of classes in TUEV dataset
        time_series_attrs = torch.randn(batch_size, self.ts_num_channels, 1000)  # (B, S, T)
        labels = torch.randint(0, num_classes, (batch_size,))
        return time_series_attrs, labels

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
        labels: Tensor,
        summary_mode: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Tokenizes the input time-series attributes, providing a mask of non-missing attributes.

        Args:
            time_series_attrs: (B, C, T), Batch of time-series datasets.
            labels: (B, 1), Batch of labels corresponding to the multivariate time-series datasets.

        Returns:
            - (Support, 1), Support set labels.
            - (Query, 1), Query set labels.
            - (Support+Query, C, T), Time series input for support and query sets.
        """
        assert time_series_attrs is not None, "At least time_series_attrs must be provided to process_data."

        # # Print value count on labels
        # print(f"Full batch label distribution: {pd.Series(labels.cpu().numpy()).value_counts().to_dict()}")
        if self.training or summary_mode:
            ts_batch_support, ts_batch_query, y_batch_support, y_batch_query = stratified_batch_split(
                data=time_series_attrs,
                labels=labels,
            )
        else:
            ts_batch_support = time_series_attrs.to(self.device)  # (Support+Query, C, T)
            ts_batch_query = time_series_attrs.to(self.device)  # (Support+Query, C, T)
            y_batch_support = labels.to(self.device)  # (Support, 1)
            y_batch_query = labels.to(self.device)  # (Query, 1)

        # # Print value count on labels
        # print(f"Support set label distribution: {pd.Series(y_batch_support.cpu().numpy()).value_counts().to_dict()}")
        # print(f"Query set label distribution: {pd.Series(y_batch_query.cpu().numpy()).value_counts().to_dict()}")

        ts_batch_support = self.time_series_convolution(ts_batch_support)  # (Support, C, T')
        ts_batch_query = self.time_series_convolution(ts_batch_query)  # (Query, C, T')
        # Flatten
        ts_batch_support = ts_batch_support.view(ts_batch_support.size(0), -1)  # (Support, C*T')
        ts_batch_query = ts_batch_query.view(ts_batch_query.size(0), -1)  # (Query, C*T')
        # if self.ts_tokenizer is not None:
        #     ts_batch_support = self.ts_tokenizer(
        #         ts_batch_support,
        #         input_chans=list(range(self.ts_num_channels)),
        #     )  # (Support, num_tokens)
        #     ts_batch_query = self.ts_tokenizer(
        #         ts_batch_query,
        #         input_chans=list(range(self.ts_num_channels)),
        #     )  # (Query, num_tokens)

        # Unsqueeze to comply with expected input shape for TabPFN encoder
        if ts_batch_support.ndim == 2:
            ts_batch_support = ts_batch_support.unsqueeze(0)  # (1, Support, C*T)
        if ts_batch_query.ndim == 2:
            ts_batch_query = ts_batch_query.unsqueeze(0)  # (1, Query, C*T)
        if y_batch_support.ndim == 1:
            y_batch_support = y_batch_support.unsqueeze(0)  # (1, Support)
        if y_batch_query.ndim == 1:
            y_batch_query = y_batch_query.unsqueeze(0)  # (1, Query)

        return (
            y_batch_support,
            y_batch_query,
            ts_batch_support,
            ts_batch_query,
        )

    def get_knn(
        self,
        y_support: Tensor,
        y_query: Tensor,
        ts_support: Tensor,
        ts_query: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Sequence[Tensor]]:

        num_classes = len(torch.unique(y_support))
        ts_support_np = ts_support.detach().cpu().numpy().copy()
        y_support_np = y_support.detach().cpu().numpy().copy()

        faissKNN = MulticlassFaiss(
            embX=ts_support_np,
            X_orig=ts_support_np,  # FIXME: remove duplicate in the class builder
            y=y_support_np,
            metric="L2",
        )
        sizes_per_class = get_sizes_per_class(
            "equal",
            y_support_np,
            num_classes=num_classes,
            context_length=ts_support.shape[1],
        )
        indices_ts_query_nni, y_nni = faissKNN.get_nearest_neighbors(
            embX_query=ts_query.cpu().numpy(),
            sizes_per_class=sizes_per_class,
        )
        ts_nni = np.concatenate(
            [ts_support_np[y_support_np == i][indices] for i, indices in enumerate(indices_ts_query_nni)],
            axis=0,
        )

    @auto_move_data
    def encode(
        self,
        y_batch_support: Tensor,
        ts_batch_support: Tensor,
        ts_batch_query: Tensor,
        y_inference_support: Optional[Tensor] = None,
        ts_inference_support: Optional[Tensor] = None,
    ) -> Tensor:
        """Embeds input sequences using the encoder model, optionally selecting/pooling output ts for the embedding.

        Args:
            y_batch_support: (S, 1), Support set labels.
            ts: (B, S (=Support+Query), T), Tokens to feed to the encoder.
        Returns: (B, Query, E), Embeddings of the input sequences.
        """
        if self.training or y_inference_support is None:
            ts = torch.cat([ts_batch_support, ts_batch_query], dim=1)  # (B, S+Q, T)
            out_features = self.encoder(
                ts.transpose(0, 1), y_batch_support.transpose(0, 1), ts_pe=self.time_series_positional_encoding
            )[:, :, -1, :]

        elif y_inference_support is not None and ts_inference_support is not None:
            # Use train set as context for predicting the query set on val/test inference
            ts = torch.cat([ts_inference_support, ts_batch_query], dim=1)
            y_train = y_inference_support
            out_features = self.encoder(
                ts.transpose(0, 1), y_train.transpose(0, 1), ts_pe=self.time_series_positional_encoding
            )[:, :, -1, :]

        else:
            raise ValueError("During inference, both support ts and labels must be provided.")

        return out_features  # (B, Query, E)

    @auto_move_data
    def forward(
        self,
        time_series_attrs: Tensor,
        labels: Tensor,
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

        if hasattr(self, "example_input_array") and torch.equal(
            time_series_attrs, self.example_input_array[0].to(time_series_attrs.device)
        ):
            summary_mode = True
        else:
            summary_mode = False

        y_batch_support, y_batch_query, ts_support, ts_query = self.process_data(
            time_series_attrs=time_series_attrs,
            labels=labels,
            summary_mode=summary_mode,
        )  # (B, Support, 1), (B, Query, 1), (B, S, T)

        out_features = self.encode(y_batch_support, ts_support, ts_query)  # (B, S, E) -> (B, E)

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

        y_batch_support, y_batch_query, ts_support, ts_query = self.process_data(
            time_series_attrs=time_series_input
        )  # (B, S, E), (B, S)
        return self.encode(
            y_batch_support,
            ts_support,
            ts_query,
        )  # (B, S, E) -> (B, E)

    def _shared_step(self, batch: Union[Tensor, Tuple[Tensor, ...]], batch_idx: int) -> Dict[str, Tensor]:
        # Shared step for training, validation and testing
        metrics = {}
        losses = []
        if self.predict_losses is not None:
            metrics.update(self._prediction_shared_step(batch, num_classes=6))  # Assuming 6 classes for TUEV dataset
            losses.append(metrics["s_loss"])

        # Compute the sum of the (weighted) losses
        metrics["loss"] = sum(losses)

        return metrics

    def _prediction_shared_step(
        self,
        batch: Union[Tensor, Tuple[Tensor, ...]],
        num_classes: int,
    ) -> Dict[str, Tensor]:
        # Forward pass through the encoder without gradient computation to fine-tune only the prediction heads
        assert (
            self.prediction_heads is not None
        ), "You requested to perform a prediction task, but the model does not include any prediction heads."
        if self.training:
            time_series_input, target_labels = batch  # (N, C, T), (N,)
            time_series_support = None
        else:
            batch_dict, _, _ = batch
            time_series_input, target_labels = batch_dict["val"]  # (N, C, T), (N,)
            time_series_support, support_labels = batch_dict["train"]  # (N, C, T), (N,)

        y_batch_support, y_batch_query, ts_support, ts_query = self.process_data(
            time_series_attrs=time_series_input, labels=target_labels
        )  # (B, Support, 1), (B, Query, 1), (B, S, T)
        # B not equal to N (dataset batch size = 1 here)

        if time_series_support is not None:
            assert support_labels is not None, "Support labels must be provided for inference."
            # Store inference data for val/test steps
            y_train_support, _, ts_train_support, _ = self.process_data(
                time_series_attrs=time_series_support, labels=support_labels
            )  # (B, Support, 1), (B, Query, 1), (B, S, T)
            y_inference_support = y_train_support
        else:
            y_inference_support = None
            ts_train_support = None

        prediction = self.encode(
            y_batch_support,
            ts_support,
            ts_query,
            y_inference_support=y_inference_support,
            ts_inference_support=ts_train_support,
        )
        predictions = {}
        for target_task, prediction_head in self.prediction_heads.items():
            pred = prediction_head(prediction)
            predictions[target_task] = pred.squeeze(dim=0).squeeze(dim=0)  # (B=Query, num_classes)

        # Compute the loss/metrics for each target label, ignoring items for which targets are missing
        losses, metrics = {}, {}

        target_batch = y_batch_query

        if self.trainer.training:
            stage = "train_metrics"
        elif self.trainer.validating:
            stage = "val_metrics"
        else:
            stage = "test_metrics"
        for target_task, target_loss in self.predict_losses.items():
            y_hat = predictions[target_task]  # (N=Query, num_classes)
            if y_hat.ndim == 1:
                y_hat = y_hat.unsqueeze(dim=0)  # (N=Query, num_classes=1)
            target = target_batch.squeeze(dim=0)  # (N=Query,)
            # Convert target to long if classification with >2 classes, float otherwise
            target = target.long()  # TODO: adapt for binary classification
            losses[f"{target_loss.__class__.__name__.lower().replace('loss', '')}/{target_task}"] = target_loss(
                y_hat,
                target,
            )

            # Metrics are automatically updated inside the Metric objects
            self.metrics[stage][target_task].update(y_hat, target)

        losses["s_loss"] = torch.stack(list(losses.values())).mean()
        metrics.update(losses)

        return metrics

    def on_test_epoch_end(self):
        output_data = []

        for target_task, collection in self.metrics["test_metrics"].items():
            # compute() returns a dict of results for this task
            results = collection.compute()

            for metric_name, value in results.items():
                tag = f"{metric_name}/{target_task}"
                self.log(f"test_{tag}", value)  # Log to logger
                output_data.append({"metric": tag, "value": value.item()})

            # Reset is handled by Lightning if logged, but manual reset is safe here
            collection.reset()

        # Save CSV once at the very end
        # df = pd.DataFrame(output_data)
        # df.to_csv("test_metrics.csv", index=False)
        with open("test_metrics.csv", mode="w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=["metric", "value"])
            writer.writeheader()
            writer.writerows(output_data)
            # for row in output_data:
            #     writer.writerow(row)
