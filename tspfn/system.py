import sys
from abc import ABC
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Union, Optional, Tuple

import hydra
import pytorch_lightning as pl
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig
import torch
from omegaconf import DictConfig
from torch import Tensor, nn
from torchinfo import summary

from data.utils.decorators import prefix_native


class TSPFNSystem(pl.LightningModule, ABC):
    """Top-level abstract system from which to inherit, implementing some generic utilities and boilerplate code.

    Implementations of behaviors related to each phase of training (e.g. data preparation, training, evaluation) should
    be made in specific packages (e.g. `data` for data handling, `tasks` for computation pipeline) or in callbacks
    (e.g. evaluation).
    """

    def __init__(self, model: DictConfig, optim: DictConfig, choices: DictConfig, **kwargs):
        """Saves the system's configuration in `hparams`.

        Args:
            model: Nested Omegaconf object containing the model architecture's configuration.
            optim: Nested Omegaconf object containing the optimizers' configuration.
            choices: Nested Omegaconf object containing the choice made from the pre-set configurations.
            data_params: Parameters related to the data necessary to initialize the model.
            **kwargs: Dictionary of arguments to save as the model's `hparams`.
        """
        super().__init__()
        # Collection of hyperparameters configuring the system
        self.save_hyperparameters()

        # Also save the classpath of the system to be able to load a checkpoint w/o knowing it's type beforehand
        self.save_hyperparameters({"task": {"_target_": f"{self.__class__.__module__}.{self.__class__.__name__}"}})

    @classmethod
    def load_from_checkpoint(cls, *args, **kwargs):  # noqa: D102
        if ABC in cls.__bases__:
            # We use this method to determine if the class is abstract because Lightning does not use the
            # `@abstractmethod` decorator to mark abstract methods (e.g. `train_step`), rather relying on runtime
            # warnings. Because of this, we cannot rely on Python's canonical function for detecting abstract classes,
            # `inspect.isabstract`, which expects abstract classes to have explicitly defined abstract methods.
            raise NotImplementedError(
                f"Class '{cls.__name__}' does not support being loaded from a checkpoint because it is an ABC. Either "
                f"call `load_from_checkpoint` on the specific class of the system you want to load, or use the utility "
                f"function `load_from_checkpoint` provided in `dataprocessing.utils.saving` that will automatically detect the "
                f"system class when loading the checkpoint."
            )
        else:
            return super().load_from_checkpoint(*args, **kwargs)

    def configure_model(self) -> Tuple[nn.Module, Optional[nn.Module], Optional[nn.ModuleDict]]:
        """Configure the network architecture used by the system."""
        return hydra.utils.instantiate(
            self.hparams["model"],
            input_shape=self.hparams["data_params"].in_shape,
            output_shape=self.hparams["data_params"]["out_shape"],
        )

    def configure_optimizers(
        self, params: Union[Iterable[Tensor], Iterable[dict]] = None
    ) -> OptimizerLRSchedulerConfig:
        """Configures optimizers/LR schedulers based on hydra config.

        Supports 2 configuration schemes:
        1) an optimizer configured alone at the root of the `optim` config node, or
        2) an optimizer configured in an `optim.optimizer` node, along with an (optional) LR scheduler configured in an
           `optim.lr_scheduler` node.

        Args:
            params: Model parameters with which to initialize the optimizer.

        Returns:
            A dict with an `optimizer` key, and an optional `lr_scheduler` if a scheduler is used.
        """
        if params is None:
            # params = self.parameters()
            params = filter(lambda p: p.requires_grad, self.parameters())

        # Extract the optimizer and scheduler configs
        # if optimizer_cfg := self.hparams["optim"].get("optimizer"):
        #     scheduler_cfg = self.hparams["optim"].get("lr_scheduler")
        # else:
        #     optimizer_cfg = self.hparams["optim"]

        # optimizer_cfg = self.hparams["optim"]["optimizer"]
        # scheduler_cfg = None
        # if scheduler_cfg := self.hparams["optim"].get("scheduler"):
        #     print("Configuring scheduler with the following config:")
        #     print(scheduler_cfg)
        #     total_steps = self.trainer.estimated_stepping_batches
        #     warmup_ratio = scheduler_cfg.get("num_warmup_steps", 0)
        #     num_warmup = int(total_steps * warmup_ratio)

        # # Instantiate the optimizer and scheduler
        # configured_optimizer = {"optimizer": hydra.utils.instantiate(optimizer_cfg, params=params)}
        # if scheduler_cfg:
        #     configured_optimizer["lr_scheduler"] = {
        #         "scheduler": hydra.utils.instantiate(
        #             scheduler_cfg,
        #             optimizer=configured_optimizer["optimizer"],
        #             num_warmup_steps=num_warmup,
        #             num_training_steps=total_steps,
        #         ),
        #         "interval": "step",
        #     }

        # return configured_optimizer
        optimizer_cfg = self.hparams["optim"]["optimizer"]
        scheduler_cfg = self.hparams["optim"].get("scheduler")

        # 1. Instanciation de l'optimiseur
        optimizer = hydra.utils.instantiate(optimizer_cfg, params=params)
        
        output = {"optimizer": optimizer}

        if scheduler_cfg:
            # Calcul dynamique des steps
            total_steps = self.trainer.estimated_stepping_batches
            
            warmup_ratio = scheduler_cfg.get("num_warmup_steps", 0.1)
            num_warmup = int(total_steps * warmup_ratio)

            print(f"Configuring scheduler: Total steps={total_steps}, Warmup steps={num_warmup}")

            sch_kwargs = dict(scheduler_cfg)
            sch_kwargs.pop("num_warmup_steps", None)
            sch_kwargs.pop("num_training_steps", None)

            scheduler = hydra.utils.instantiate(
                sch_kwargs,
                optimizer=optimizer,
                num_warmup_steps=num_warmup,
                num_training_steps=total_steps,
            )

            output["lr_scheduler"] = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }

        return output

    def _shared_step(self, *args, **kwargs) -> Dict[str, Tensor]:
        """Handles steps for the train/val/test loops, assuming the behavior should be the same.

        Returns:
            Mapping between metric names and their values. It must contain at least a ``'loss'``, as that is the value
            optimized in training and monitored by callbacks during validation.
        """
        raise NotImplementedError

    def training_step(self, *args, **kwargs) -> Dict[str, Tensor]:  # noqa: D102
        result = prefix_native(self._shared_step(*args, **kwargs), "train/")
        self.log_dict(result, **self.hparams.train_log_kwargs)
        # Add reference to 'train_loss' under 'loss' keyword, requested by PL to know which metric to optimize
        result["loss"] = result["train/loss"]
        return result

    def validation_step(self, *args, **kwargs) -> Dict[str, Tensor]:  # noqa: D102
        result = prefix_native(self._shared_step(*args, **kwargs), "val/")
        self.log_dict(result, **self.hparams.val_log_kwargs)
        return result

    def test_step(self, *args, **kwargs) -> Dict[str, Tensor]:  # noqa: D102
        result = prefix_native(self._shared_step(*args, **kwargs), "test/")
        self.log_dict(result, **self.hparams.val_log_kwargs)
        return result

    def setup(self, stage: str) -> None:  # noqa: D102
        self.log_dir.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists

        # Save Keras-style summary to a ``summary.txt`` file, inside the output directory
        # Option to disable the summary if PL model's summary is disabled to avoid possible device incompatibilities
        # (e.g. in clusters).
        if stage == TrainerFn.FITTING and self.hparams["enable_model_summary"] and self.global_rank == 0:
            model_summary = summary(
                self,
                input_data=self.example_input_array,
                col_names=["input_size", "output_size", "kernel_size", "num_params"],
                depth=sys.maxsize,
                device=self.device,
                mode=self.hparams["model_summary_mode"],
                verbose=0,
            )
            (self.log_dir / "summary.txt").write_text(str(model_summary), encoding="utf-8")

    @property
    def log_dir(self) -> Path:
        """Returns the root directory where test logs get saved."""
        return Path(self.trainer.log_dir) if self.trainer.log_dir else Path(self.trainer.default_root_dir)
