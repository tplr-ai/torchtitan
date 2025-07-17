# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torchtitan.experiments.simple_fsdp  # noqa: F401

import importlib, warnings

try:
    importlib.import_module(f"torchtitan.models.llama4")  # noqa: F401
except ImportError as e:
    warnings.warn(f"Skipping llama4: {e}")
