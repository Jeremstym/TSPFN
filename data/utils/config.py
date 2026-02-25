import builtins
import logging
import operator
import os
from configparser import ConfigParser
from pathlib import Path
from typing import Any, Callable, List

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning import Callback

logger = logging.getLogger(__name__)


def register_omegaconf_resolvers() -> None:
    """Registers various OmegaConf resolvers useful to query system/repository/config info."""
    OmegaConf.register_new_resolver("sys.num_workers", lambda x=None: os.cpu_count() - 1, replace=True)
    OmegaConf.register_new_resolver("sys.num_gpus", lambda x=None: torch.cuda.device_count(), replace=True)
    OmegaConf.register_new_resolver("sys.getcwd", lambda x=None: os.getcwd(), replace=True)
    OmegaConf.register_new_resolver("sys.eps.np", lambda dtype: np.finfo(np.dtype(dtype)).eps, replace=True)

    # Define wrapper for basic math operators, with the option to cast result to arbitrary type
    def _cast_op(op, x, y, type_of: str = None) -> Any:
        res = op(x, y)
        if type_of is not None:
            res = getattr(builtins, type_of)(res)
        return res

    OmegaConf.register_new_resolver("op.add", lambda x, y, type_of=None: _cast_op(operator.add, x, y, type_of=type_of))
    OmegaConf.register_new_resolver("op.sub", lambda x, y, type_of=None: _cast_op(operator.sub, x, y, type_of=type_of))
    OmegaConf.register_new_resolver("op.mul", lambda x, y, type_of=None: _cast_op(operator.mul, x, y, type_of=type_of))
    OmegaConf.register_new_resolver("op.mod", lambda x, y, type_of=None: _cast_op(operator.mod, x, y, type_of=type_of))
    OmegaConf.register_new_resolver(
        "op.cat", lambda x, y, type_of=None: _cast_op(operator.concat, x, y, type_of=type_of)
    )

    OmegaConf.register_new_resolver("builtin.len", lambda cfg: len(cfg))
    OmegaConf.register_new_resolver("builtin.range", lambda start, stop, step=1: list(range(start, stop, step)))
    OmegaConf.register_new_resolver(
        "list.remove",
        lambda cfg, to_remove: ListConfig(
            [
                val
                for val in cfg
                if (val not in to_remove if isinstance(to_remove, (tuple, list, ListConfig)) else val != to_remove)
            ]
        ),
    )
    OmegaConf.register_new_resolver("list.at", lambda cfg, idx: cfg[idx])

    def resolve_tuple(*args):
        return tuple(args)

    OmegaConf.register_new_resolver("tuple", resolve_tuple)


def instantiate_config_node_leaves(
    cfg: DictConfig, node_desc: str, instantiate_fn: Callable[[DictConfig, str], Any] = None
) -> List[Any]:
    """Iterates over the leafs of the `cfg` config node and instantiates the leaves.

    Args:
        cfg: Root node whose leaves are to be instantiated.
        node_desc: Description of the node, used to display relevant messages in the logs. If not provided,
            defaults to `node_name`.
        instantiate_fn: Callback that instantiates an object from the config. If not provided, will default to call
            `hydra.utils.instantiate`.

    Returns:
        Objects instantiated from the leaves of the `cfg` config node.
    """
    if not instantiate_fn:
        instantiate_fn = hydra.utils.instantiate

    objects = []
    for obj_name, obj_cfg in cfg.items():
        if "_target_" in obj_cfg:
            logger.info(f"Instantiating {node_desc} <{obj_name}>")
            instantiate_args = []
            if instantiate_fn != hydra.utils.instantiate:  # If using a custom instantiation function
                instantiate_args = [obj_name]
            objects.append(instantiate_fn(obj_cfg, *instantiate_args))
        else:
            logger.warning(f"No '_target_' field in {node_desc} config. Cannot instantiate {obj_name}")
    return objects
