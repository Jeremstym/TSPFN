import functools
import importlib
import itertools
import logging
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Any, Callable, Dict, Literal, Optional, Sequence, Tuple, cast

import hydra
import torch
from dataprocessing.data.orchid.config import OrchidTag, TimeSeriesAttribute
from dataprocessing.data.orchid.config import View as ViewEnum
from dataprocessing.data.orchid.datapipes import (
    PatientData,
    PatientDataTarget,
    filter_time_series_attributes,
)
from dataprocessing.tasks.generic import SharedStepsTask
from dataprocessing.utils.decorators import auto_move_data
from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig

# import dataprocessing
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.nn import Parameter, ParameterDict, init
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy, auroc, average_precision, f1_score, mean_absolute_error
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

from didactic.utils.compliance import check_model_encoder

import didactic.models.transformer
import didactic.models.fusionners
import didactic.models.transformer

logger = logging.getLogger(__name__)

torch.set_printoptions(threshold=10000)


class CardiacMultimodalTabPFN(pl.LightningModule):
    """Multi-modal transformer to learn a representation from cardiac imaging and patient records data."""

    def __init__(
        self,
        embed_dim: int,
        time_series_attrs: Sequence[TimeSeriesAttribute],
        views: Sequence[ViewEnum] = tuple(ViewEnum),
        predict_losses: Optional[Dict[str, Callable[[Tensor, Tensor], Tensor]] | DictConfig] = None,
        *args,
        **kwargs,
    ):
        """Initializes class instance.

        Args:
            embed_dim: Size of the tokens/embedding for all the modalities.
            time_series_attrs: Time-series attributes to provide to the model.
            views: Views from which to include time-series attributes.
            predict_losses: Supervised criteria to measure the error between the predicted attributes and their real
                value.
            *args: Positional arguments to pass to the parent's constructor.
            **kwargs: Keyword arguments to pass to the parent's constructor.
        """
        # Ensure string tags are converted to their appropriate enum types
        # And do it before call to the parent's `init` so that the converted values are saved in `hparams`
        views = tuple(ViewEnum[e] for e in views)
        time_series_attrs = tuple(TimeSeriesAttribute[e] for e in time_series_attrs)

        print(f"number of time series attributes: {len(time_series_attrs)}")
        print(f"number of views: {len(views)}")

        super().__init__(*args, **kwargs)

        # Add shortcut to lr to work with Lightning's learning rate finder
        self.hparams["lr"] = None

        # Add shortcut to token labels to avoid downstream applications having to determine them from hyperparameters
        self.token_tags = (
            tuple("/".join([view, attr]) for view, attr in itertools.product(views, time_series_attrs))
        )

        # Configure losses/metrics to compute at each train/val/test step
        self.metrics = nn.ModuleDict()

        # Supervised losses and metrics
        self.predict_losses = {}
        if predict_losses:
            self.predict_losses = {
               attr: (  # type: ignore[misc]
                    hydra.utils.instantiate(attr_loss) if isinstance(attr_loss, DictConfig) else attr_loss
                )
                for attr, attr_loss in predict_losses.items()
            }
        
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
        num_classes = 10 #TODO: adapt to variable number of classes
        self.metrics["classification"] = MetricCollection(
            [
                MulticlassAccuracy(num_classes=num_classes, average="micro"),
                MulticlassAUROC(num_classes=num_classes, average="macro"),
                MulticlassAveragePrecision(num_classes=num_classes, average="macro"),
                MulticlassF1Score(num_classes=num_classes, average="macro"),
            ]
        )

        # Compute shapes relevant for defining the models' architectures
        self.n_time_series_attrs = len(self.hparams["time_series_attrs"]) * len(self.hparams["views"])
        self.sequence_length = self.n_time_series_attrs

        # Self-supervised losses and metrics
        # Initialize transformer encoder and self-supervised + prediction heads
        self.encoder, self.prediction_heads = self.configure_model()

        # Configure tokenizers and extract relevant info about the models' architectures
        self.nhead, self.separate_modality = check_model_encoder(self.encoder, self.hparams)

        self.mask_token = None

        # Initialize inference storage tensors
        self.ts_train_for_inference = torch.Tensor().to(self.device)
        self.y_train_for_inference = torch.Tensor().to(self.device)

    def on_validation_epoch_start(self):
        # Assumes the first dataloader is the train loader
        train_loader = self.trainer.train_dataloader
        if train_loader is None:
            print("Validation sanity check")
        else:
            if callable(train_loader):
                train_loader = train_loader()

            ts_attrs_for_train_inference = {}
            for batch in train_loader:
                ts_for_train_inference = {
                    attr: attr_data
                    for attr, attr_data in filter_time_series_attributes(
                        batch, views=self.hparams["views"], attrs=self.hparams["time_series_attrs"]
                    )[0].items()
                }
                for attr, view in ts_for_train_inference.keys():
                    ts_attrs_for_train_inference.setdefault((attr, view), []).append(
                        ts_for_train_inference[(attr, view)]
                    )

            ts_cat_attrs_for_train_inference = {
                attr: torch.cat(ts_attrs_for_train_inference[attr], dim=0) for attr in ts_attrs_for_train_inference
            }
            ts_data: Dict[Tuple[str, str], List[Tensor]] = {}
            for cross_attrs in ts_cat_attrs_for_train_inference:
                cross_attrs_column: Tuple[str, str] = (cross_attrs[0].__str__(), cross_attrs[1].__str__())
                ts_data_value = F.interpolate(
                    ts_cat_attrs_for_train_inference[cross_attrs].unsqueeze(1), size=64, mode="linear"
                ).squeeze(1)
                ts_data.setdefault(cross_attrs_column, []).append(ts_data_value)
            ts_data_stacked = {
                ts_view_attr: torch.vstack(batch_vals) for ts_view_attr, batch_vals in ts_data.items()
            }
            ts_data_doublestack = {
                ts_view: torch.hstack(
                    [ts_data_stacked[(view, attr)] for (view, attr) in ts_data_stacked.keys() if view == ts_view]
                )
                for ts_view in self.hparams["views"]
            }
            ts_tokens_validation_list = []
            for ts_view in ts_data_doublestack:
                ts_view_data = ts_data_doublestack[ts_view]
                ts_tokens_validation_list.append(ts_view_data)
            ts_tokens_validation = torch.stack(
                ts_tokens_validation_list, dim=1
            ).float()  # (N, 1, S_ts * num_time_units)
            self.ts_train_for_inference = ts_tokens_validation.to(self.device)

    def on_test_epoch_start(self):
        train_loader = self.trainer.test_dataloaders[1]
        if train_loader is None:
            raise ValueError("No training dataloader found while setting up inference storage tensors.")
        else:
            if callable(train_loader):
                train_loader = train_loader()

            ts_attrs_for_train_inference = {}
            for batch in train_loader:
                ts_for_train_inference = {
                    attr: attr_data
                    for attr, attr_data in filter_time_series_attributes(
                        batch, views=self.hparams["views"], attrs=self.hparams["time_series_attrs"]
                    )[0].items()
                }
                for attr, view in ts_for_train_inference.keys():
                    ts_attrs_for_train_inference.setdefault((attr, view), []).append(
                        ts_for_train_inference[(attr, view)]
                    )

            ts_cat_attrs_for_train_inference = {
                attr: torch.cat(ts_attrs_for_train_inference[attr], dim=0) for attr in ts_attrs_for_train_inference
            }
            ts_data: Dict[Tuple[str, str], List[Tensor]] = {}
            for cross_attrs in ts_cat_attrs_for_train_inference:
                cross_attrs_column: Tuple[str, str] = (cross_attrs[0].__str__(), cross_attrs[1].__str__())
                ts_data_value = F.interpolate(
                    ts_cat_attrs_for_train_inference[cross_attrs].unsqueeze(1), size=64, mode="linear"
                ).squeeze(1)
                ts_data.setdefault(cross_attrs_column, []).append(ts_data_value)
            ts_data_stacked = {
                ts_view_attr: torch.vstack(batch_vals) for ts_view_attr, batch_vals in ts_data.items()
            }
            ts_data_doublestack = {
                ts_view: torch.hstack(
                    [ts_data_stacked[(view, attr)] for (view, attr) in ts_data_stacked.keys() if view == ts_view]
                )
                for ts_view in self.hparams["views"]
            }
            ts_tokens_test_list = []
            for ts_view in ts_data_doublestack:
                ts_view_data = ts_data_doublestack[ts_view]
                ts_tokens_test_list.append(ts_view_data)
            ts_tokens_test = torch.stack(ts_tokens_test_list, dim=1).float()  # (N, 1, S_ts * num_time_units)
            self.ts_train_for_inference = ts_tokens_test.to(self.device)

    def on_shared_epoch_start(self):
        train_loader = self.trainer.train_dataloader
        if train_loader is None:
            raise ValueError("No training dataloader found while setting up inference storage tensors.")
        else:
            if callable(train_loader):
                train_loader = train_loader()

            ts_attrs_for_train_inference = {}
            for batch in train_loader:
                ts_for_train_inference = {
                    attr: attr_data
                    for attr, attr_data in filter_time_series_attributes(
                        batch, views=self.hparams["views"], attrs=self.hparams["time_series_attrs"]
                    )[0].items()
                }
                for attr, view in ts_for_train_inference.keys():
                    ts_attrs_for_train_inference.setdefault((attr, view), []).append(
                        ts_for_train_inference[(attr, view)]
                    )

            ts_cat_attrs_for_train_inference = {
                attr: torch.cat(ts_attrs_for_train_inference[attr], dim=0) for attr in ts_attrs_for_train_inference
            }
            ts_data: Dict[Tuple[str, str], List[Tensor]] = {}
            for cross_attrs in ts_cat_attrs_for_train_inference:
                cross_attrs_column: Tuple[str, str] = (cross_attrs[0].__str__(), cross_attrs[1].__str__())
                ts_data_value = F.interpolate(
                    ts_cat_attrs_for_train_inference[cross_attrs].unsqueeze(1), size=64, mode="linear"
                ).squeeze(1)
                ts_data.setdefault(cross_attrs_column, []).append(ts_data_value)
            ts_data_stacked = {
                ts_view_attr: torch.vstack(batch_vals) for ts_view_attr, batch_vals in ts_data.items()
            }
            ts_data_doublestack = {
                ts_view: torch.hstack(
                    [ts_data_stacked[(view, attr)] for (view, attr) in ts_data_stacked.keys() if view == ts_view]
                )
                for ts_view in self.hparams["views"]
            }
            ts_tokens_shered_list = []
            for ts_view in ts_data_doublestack:
                ts_view_data = ts_data_doublestack[ts_view]
                ts_tokens_shered_list.append(ts_view_data)
            ts_tokens_shered = torch.stack(
                ts_tokens_shered_list, dim=1
            ).float()  # (N,

    def on_predict_epoch_start(self):
        train_loader = self.trainer.predict_dataloaders[1]
        if train_loader is None:
            raise ValueError("No training dataloader found while setting up inference storage tensors.")
        else:
            if callable(train_loader):
                train_loader = train_loader()

            ts_attrs_for_train_inference = {}
            for batch in train_loader:
                ts_for_train_inference = {
                    attr: attr_data
                    for attr, attr_data in filter_time_series_attributes(
                        batch, views=self.hparams["views"], attrs=self.hparams["time_series_attrs"]
                    )[0].items()
                }
                for attr, view in ts_for_train_inference.keys():
                    ts_attrs_for_train_inference.setdefault((attr, view), []).append(
                        ts_for_train_inference[(attr, view)]
                    )

            ts_cat_attrs_for_train_inference = {
                attr: torch.cat(ts_attrs_for_train_inference[attr], dim=0) for attr in ts_attrs_for_train_inference
            }
            ts_data: Dict[Tuple[str, str], List[Tensor]] = {}
            for cross_attrs in ts_cat_attrs_for_train_inference:
                cross_attrs_column: Tuple[str, str] = (cross_attrs[0].__str__(), cross_attrs[1].__str__())
                ts_data_value = F.interpolate(
                    ts_cat_attrs_for_train_inference[cross_attrs].unsqueeze(1), size=64, mode="linear"
                ).squeeze(1)
                ts_data.setdefault(cross_attrs_column, []).append(ts_data_value)
            ts_data_stacked = {
                ts_view_attr: torch.vstack(batch_vals) for ts_view_attr, batch_vals in ts_data.items()
            }
            ts_data_doublestack = {
                ts_view: torch.hstack(
                    [ts_data_stacked[(view, attr)] for (view, attr) in ts_data_stacked.keys() if view == ts_view]
                )
                for ts_view in self.hparams["views"]
            }
            ts_tokens_prediction_list = []
            for ts_view in ts_data_doublestack:
                ts_view_data = ts_data_doublestack[ts_view]
                ts_tokens_prediction_list.append(ts_view_data)
            ts_tokens_prediction = torch.stack(
                ts_tokens_prediction_list, dim=1
            ).float()  # (N, 1, S_ts * num_time_units)
            self.ts_train_for_inference = ts_tokens_prediction.to(self.device)

    @property
    def example_input_array(
        self,
    ) -> Dict[Tuple[ViewEnum, TimeSeriesAttribute], Tensor]:
        """Redefine example input array based on the cardiac attributes provided to the model."""
        # 2 is the size of the batch in the example
        labels = torch.randperm(10)
        time_series_attrs = {
            (view, attr): torch.randn(10, self.hparams["data_params"].in_shape[OrchidTag.time_series_attrs][1])
            for view, attr in itertools.product(self.hparams["views"], self.hparams["time_series_attrs"])
        }
        # time_series_notna_mask = torch.ones(
        #     (10, len(self.hparams["views"]) * len(self.hparams["time_series_attrs"])), dtype=torch.bool
        # )
        return time_series_attrs  # , time_series_notna_mask

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
            output_size = 10 # follows TabPFN's default setting for classification tasks
            prediction_heads["classification"] = hydra.utils.instantiate(
                self.hparams["model"]["prediction_head"], out_features=output_size
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
        time_series_attrs: Dict[Tuple[ViewEnum, TimeSeriesAttribute], Tensor],
        target: Tensor,
        # time_series_notna_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Tokenizes the input time-series attributes, providing a mask of non-missing attributes.

        Args:
            time_series_attrs: (K: S, V: (N, ?)), Sequence of batches of time-series attributes, where the
                dimensionality of each attribute can vary.

        Returns:
            Batch of i) (N, S, E) tokens for each attribute, and ii) (N, S) mask of non-missing attributes.
        """
        # Initialize lists for cumulating (optional) tensors for each modality, that will be concatenated into tensors
        ts_tokens_list, notna_mask_list = [], []

        # Tokenize the attributes
        assert time_series_attrs is not None, "At least time_series_attrs must be provided to process_data."
        ts_data: Dict[Tuple[str, str], List[Tensor]] = {}
        for cross_attrs in time_series_attrs:
            cross_attrs_column: Tuple[str, str] = (cross_attrs[0].__str__(), cross_attrs[1].__str__())
            ts_data_value = F.interpolate(time_series_attrs[cross_attrs].unsqueeze(1), size=64, mode="linear").squeeze(
                1
            )
            ts_data.setdefault(cross_attrs_column, []).append(ts_data_value)
        ts_data_stacked = {ts_view_attr: torch.vstack(batch_vals) for ts_view_attr, batch_vals in ts_data.items()}
        ts_data_doublestack = {
            ts_view: torch.hstack(
                [ts_data_stacked[(view, attr)] for (view, attr) in ts_data_stacked.keys() if view == ts_view]
            )
            for ts_view in self.hparams["views"]
        }
        for ts_view in ts_data_doublestack:
            ts_view_data = ts_data_doublestack[ts_view]
            ts_tokens_list.append(ts_view_data)

        ts_tokens = torch.stack(ts_tokens_list, dim=1)  # (N, 1, S_ts * num_time_units)

        # Cast to float to make sure tokens are not represented using double
        ts_tokens = ts_tokens.float()

        # Indicate that, when time-series tokens are requested, they are always available
        # if time_series_notna_mask is None:
        #     time_series_notna_mask = torch.full(ts_tokens.shape[:2], True, device=ts_tokens.device)

        indices = list(range(len(ts_tokens)))
        y = target
        if self.training or len(self.y_train_for_inference) == 0:
            assert self.hparams["split_finetuning"] > 0.0, "split_finetuning must be > 0.0 when training."
            label = y.clone().cpu()
            try:
                train_indices, test_indices = train_test_split(
                    indices,
                    test_size=self.hparams["split_finetuning"],
                    random_state=self.hparams["seed"],
                    stratify=label,
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

            ts_tokens_support = torch.as_tensor(ts_tokens[train_indices], dtype=torch.float32)
            ts_tokens_query = torch.as_tensor(ts_tokens[test_indices], dtype=torch.float32)
            ts_tokens_support = ts_tokens_support.unsqueeze(0) if ts_tokens_support.ndim == 2 else ts_tokens_support
            ts_tokens_query = ts_tokens_query.unsqueeze(0) if ts_tokens_query.ndim == 2 else ts_tokens_query

            ts_tokens = torch.cat([ts_tokens_support, ts_tokens_query], dim=0)

        # notna_mask_list.extend(time_series_notna_mask)

        if self.training or len(self.y_train_for_inference) == 0:
            y_batch_support = torch.as_tensor(y[train_indices], dtype=torch.float32)
            y_batch_query = torch.as_tensor(y[test_indices], dtype=torch.float32)
        else:
            y_batch_support = y.to(self.device)
            y_batch_query = y.to(self.device)

        if ts_tokens.ndim == 2:
            ts_tokens = ts_tokens.unsqueeze(0)

        return (
            y_batch_support,
            y_batch_query,
            ts_tokens,
            # notna_mask,
        )

    @auto_move_data
    def encode(
        self,
        y_batch_support: Tensor,
        ts_tokens: Tensor,
        # avail_mask: Tensor,
        enable_augments: bool = False,
    ) -> Tensor:
        """Embeds input sequences using the encoder model, optionally selecting/pooling output ts_tokens for the embedding.

        Args:
            ts_tokens: (N, S, E), Tokens to feed to the encoder.
            avail_mask: (N, S), Boolean mask indicating available (i.e. non-missing) ts_tokens. Missing ts_tokens can thus be
                treated distinctly from others (e.g. replaced w/ a specific mask).
            enable_augments: Whether to perform augments on the ts_tokens (e.g. masking) to obtain a "corrupted" view for
                contrastive learning. Augments are already configured differently for training/testing (to avoid
                stochastic test-time predictions), so this parameter is simply useful to easily toggle augments on/off
                to obtain contrasting views.

        Returns: (N, E), Embeddings of the input sequences.
        """

        if self.training or len(self.ts_train_for_inference) == 0:
            out_features = self.encoder(ts_tokens, y_batch_support)[:, -1, :]
        else:
            # Use train set as context for predicting the query set
            ts_tokens = torch.cat([self.ts_train_for_inference, ts_tokens], dim=0)
            y_train = self.y_train_for_inference
            out_features = self.encoder(ts_tokens, y_train)[:, -1, :]

        if self.secondary_encoder is not None:
            out_features = self.secondary_encoder(out_features)
        return out_features  # (N, d_model)

    @auto_move_data
    def forward(
        self,
        time_series_attrs: Dict[Tuple[ViewEnum, TimeSeriesAttribute], Tensor],
        # time_series_notna_mask: Optional[Tensor],
        task: Literal["encode", "predict"] = "encode",
    ) -> Tensor | Dict[str, Tensor]:
        """Performs a forward pass through i) the tokenizer, ii) the transformer encoder and iii) the prediction head.

        Args:
            time_series_attrs: (K: S, V: (N, ?)), Sequence of batches of time-series attributes, where the
                dimensionality of each attribute can vary.
            task: Flag indicating which type of inference task to perform.

        Returns:
            if `task` == 'encode':
                (N, E), Batch of features extracted by the encoder.
            if `task` == 'continuum_param`:
                ? * (M), Parameter of the unimodal logits distribution for targets.
            if `task` == 'continuum_tau`:
                ? * (M), Temperature used to control the sharpness of the unimodal logits distribution for targets.
            if `task` == 'predict' (and the model includes prediction heads):
                ? * (N), Prediction for each target in `losses`.
        """
        if task != "encode" and not self.prediction_heads:
            raise ValueError(
                "You requested to perform a prediction task, but the model does not include any prediction heads."
            )
        time_series_attrs = {
            attr: attr_data.unsqueeze(0) if attr_data.ndim == 1 else attr_data
            for attr, attr_data in time_series_attrs.items()
        }
        # if time_series_notna_mask is not None:
        #     time_series_notna_mask = (
        #         time_series_notna_mask.unsqueeze(0) if time_series_notna_mask.ndim == 1 else time_series_notna_mask
        #     )
        y_batch_support, y_batch_query, ts_tokens = self.process_data(
            time_series_attrs,
            # time_series_notna_mask,
        )  # (N, S, E), (N, S)

        out_features = self.encode(y_batch_support, ts_tokens)  # (N, S, E) -> (N, E)

        # Early return if requested task requires no prediction heads
        if task == "encode":
            return out_features
        
        elif task == "predict":
            assert (
                self.prediction_heads is not None
            ), "You requested to perform a prediction task, but the model does not include any prediction heads."
            
            # Forward pass through each target's prediction head
            predictions = {
                attr: prediction_head(out_features) for attr, prediction_head in self.prediction_heads.items()
            }

            # Squeeze out the singleton dimension from the predictions' features (only relevant for scalar predictions)
            predictions = {attr: prediction.squeeze(dim=1) for attr, prediction in predictions.items()}
            return predictions
        else:
            raise ValueError(f"Unknown task '{task}' requested for forward pass.")

    @auto_move_data
    def get_latent_vectors(
        self,
        batch: PatientData,
        batch_idx: int,
        time_series_attrs: Dict[Tuple[ViewEnum, TimeSeriesAttribute], Tensor],
    ) -> Tensor:
        """Extracts the latent vectors from the encoder for the given batch."""
        y_batch_support, y_batch_query, ts_tokens = self.process_data(time_series_attrs)  # (N, S, E), (N, S)
        return self.encode(
            y_batch_support,
            y_batch_query,
            ts_tokens,
        )  # (N, S, E) -> (N, E)

    def _shared_step(self, batch: PatientDataTarget, batch_idx: int, dataloader_idx: int = 0) -> Dict[str, Tensor]:
        # Extract time-series attributes from the batch
        if dataloader_idx > 0 and not self.training:
            return {}
        time_series_attrs = {
            attr: attr_data.unsqueeze(0) if attr_data.ndim == 1 else attr_data
            for attr, attr_data in filter_time_series_attributes(
                batch, views=self.hparams.views, attrs=self.hparams.time_series_attrs
            )[0].items()
        }
        # time_series_notna_mask = filter_time_series_attributes(
        #     batch, views=self.hparams.views, attrs=self.hparams.time_series_attrs
        # )[1]
        # if time_series_notna_mask is not None:
        #     time_series_notna_mask = (
        #         time_series_notna_mask.unsqueeze(0) if time_series_notna_mask.ndim == 1 else time_series_notna_mask
        #     )

        # print(f'time_series_attrs: {time_series_attrs}')
        # print(f"batch id: {batch['id']}")
        # print(f'time_series_notna_mask: {time_series_notna_mask}')

        y_batch_support, y_batch_query, ts_tokens = self.process_data(time_series_attrs)  # (N, S, E), (N, S)

        metrics = {}
        losses = []
        if self.predict_losses is not None:
            metrics.update(self._prediction_shared_step(y_batch_support, y_batch_query, batch, batch_idx, ts_tokens))
            losses.append(metrics["s_loss"])

        # Compute the sum of the (weighted) losses
        metrics["loss"] = sum(losses)

        return metrics

    def _prediction_shared_step(
        self,
        y_batch_support: Tensor,
        y_batch_query: Tensor,
        batch: PatientDataTarget,
        batch_idx: int,
        ts_tokens: Tensor,
    ) -> Dict[str, Tensor]:
        # Forward pass through the encoder without gradient computation to fine-tune only the prediction heads
        assert (
            self.prediction_heads is not None
        ), "You requested to perform a prediction task, but the model does not include any prediction heads."
        prediction = self.encode(y_batch_support, ts_tokens)
        predictions = {}
        for attr, prediction_head in self.prediction_heads.items():
            pred = prediction_head(prediction)
            predictions[attr] = pred

        # Compute the loss/metrics for each target attribute, ignoring items for which targets are missing
        losses, metrics = {}, {}

        target_batch = y_batch_query

        for attr, loss in self.predict_losses.items():
            target, y_hat = target_batch, predictions[attr]

            target = target.float() if len(torch.unique(target)) == 2 else target.long()

            losses[f"{loss.__class__.__name__.lower().replace('loss', '')}/{attr}"] = loss(
                y_hat,
                target,
            )
            for metric_tag, metric in self.metrics[attr].items():
                metric.update(y_hat, target)

        losses["s_loss"] = torch.stack(list(losses.values())).mean()
        metrics.update(losses)

        return metrics

    def on_test_epoch_end(self):
        all_metrics = {}
        for attr in self.predict_losses:
            for metric_tag, metric in self.metrics[attr].items():
                metrics_value = metric.compute()
                self.log(f"test_{metric_tag}/{attr}", metrics_value)
                all_metrics[f"{metric_tag}/{attr}"] = (
                    metrics_value.item() if hasattr(metrics_value, "item") else metrics_value
                )
                metric.reset()

        # Print metrics to terminal
        logger.info(f"Test metrics: {all_metrics}")

    @torch.inference_mode()
    def predict_step(self, batch: PatientData, batch_idx: int, dataloader_idx: int = 0) -> Tuple[  # noqa: D102
        Tensor,
        Optional[Dict[str, Tensor]],
    ]:
        # print(f"time series attrs: {time_series_attrs}")
        # print(f"batch id: {batch['id']}")
        # print(f"time series notna mask: {time_series_notna_mask}")

        time_series_attrs = {
            attr: attr_data.unsqueeze(0) if attr_data.ndim == 1 else attr_data
            for attr, attr_data in filter_time_series_attributes(
                batch, views=self.hparams.views, attrs=self.hparams.time_series_attrs
            )[0].items()
        }
        # time_series_notna_mask = filter_time_series_attributes(
        #     batch, views=self.hparams.views, attrs=self.hparams.time_series_attrs
        # )[1]
        # if time_series_notna_mask is not None:
        #     time_series_notna_mask = (
        #         time_series_notna_mask.unsqueeze(0) if time_series_notna_mask.ndim == 1 else time_series_notna_mask
        #     )
        # Encoder's output
        out_features = self(time_series_attrs)

        # Remove unnecessary batch dimension from the different outputs
        # (only do this once all downstream inferences have been performed)
        out_features = out_features.squeeze(dim=0)

        # If the model has targets to predict, output the predictions
        predictions = None
        if self.prediction_heads:
            predictions = self(time_series_attrs, task="predict")

        if predictions is not None:
            predictions = {
                attr: prediction.unsqueeze(dim=0) if prediction.ndim == 1 else prediction
                for attr, prediction in predictions.items()
            }

        # Squeeze pred before returning
        if predictions is not None:
            predictions = {attr: prediction.squeeze(dim=0) for attr, prediction in predictions.items()}
        return out_features, predictions
