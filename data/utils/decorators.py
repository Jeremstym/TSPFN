from functools import wraps
from typing import Any, Callable, Dict, Mapping, Sequence, Union

import numpy as np
import torch
from torch import Tensor
from pytorch_lightning.utilities import move_data_to_device


def _has_method(o: object, name: str) -> bool:
    return callable(getattr(o, name, None))

def prefix_native(map: Mapping[str, Any], prefix: str, exclude: Union[str, Sequence[str]] = None) -> Dict[str, Any]:
    """Prepends a prefix to the keys of a mapping with string keys.

    Args:
        map: Mapping with string keys for which to add a prefix to the keys.
        prefix: Prefix to add to the current keys in the mapping.
        exclude: Keys to exclude from the prefix addition. These will remain unchanged in the new mapping.

    Returns:
        Mapping where the keys have been prepended with `prefix`.
    """
    if exclude is None:
        exclude = []
    elif isinstance(exclude, str):
        exclude = [exclude]

    return {f"{prefix}{k}" if k not in exclude else k: v for k, v in map.items()}

def prefix(
    prefix: str, exclude: Union[str, Sequence[str]] = None
) -> Callable[[Callable[..., Mapping[str, Any]]], Callable[..., Dict[str, Any]]]:
    """Decorator for functions that return a mapping with string keys, to add a prefix to the keys.

    Args:
        prefix: Prefix to add to the current keys in the mapping.
        exclude: Keys to exclude from the prefix addition. These will remain unchanged in the new mapping.

    Returns:
        Function where the keys of the mapping returned have been prepended with `prefix`, except for the keys listed in
        `exclude`, that are left as-is.
    """

    def prefix_decorator(fn: Callable[..., Mapping[str, Any]]) -> Callable[..., Dict[str, Any]]:
        @wraps(fn)
        def prefix_wrapper(self, *args, **kwargs):
            return prefix_native(fn(self, *args, **kwargs), prefix, exclude=exclude)

        return prefix_wrapper

    return prefix_decorator


def auto_move_data(fn: Callable) -> Callable:
    """Decorator for ``LightningModule`` methods for which input args should be moved to the correct device.

    Typically, applied to ``__call__`` or ``forward``.

    Args:
        fn: A LightningModule method for which the arguments should be moved to the device the parameters are on.

    Example::

        # directly in the source code
        class LitModel(LightningModule):

            @auto_move_data
            def forward(self, x):
                return x

        # or outside
        LitModel.forward = auto_move_data(LitModel.forward)

        model = LitModel()
        model = model.to('cuda')
        model(torch.zeros(1, 3))

        # input gets moved to device
        # tensor([[0., 0., 0.]], device='cuda:0')

    """

    @wraps(fn)
    def auto_transfer_args(self, *args, **kwargs):
        args, kwargs = move_data_to_device((args, kwargs), self.device)
        return fn(self, *args, **kwargs)

    return auto_transfer_args
