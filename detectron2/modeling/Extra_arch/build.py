# Copyright (c) Facebook, Inc. and its affiliates.
import torch

from detectron2.utils.logger import _log_api_usage
from detectron2.utils.registry import Registry

Extra_ARCH_REGISTRY = Registry("Extra_ARCH")  # noqa F401 isort:skip
Extra_ARCH_REGISTRY.__doc__ = """
Registry for meta-architectures, i.e. the whole model.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""


def build_extra_head(cfg):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    extra_arch = cfg.Extra_ARCH
    # print(extra_arch)
    model = Extra_ARCH_REGISTRY.get(extra_arch)(cfg)
    # print(model)
    # model.to(torch.device(cfg.MODEL.DEVICE))
    # _log_api_usage("modeling.extra_arch." + extra_arch)
    return model
