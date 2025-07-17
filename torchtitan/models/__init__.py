# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# Import the built-in models here so that the corresponding register_model_spec()
# will be called.

import torchtitan.models.llama3  # noqa: F401

import importlib, warnings

try:
    importlib.import_module(f"torchtitan.models.deepseek_v3")  # noqa: F401
except ImportError as e:
    warnings.warn(f"Skipping deepseek_v3: {e}")