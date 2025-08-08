# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# Import the built-in models here so that the corresponding register_model_spec()
# will be called.

import importlib
import warnings

import torchtitan.models.llama3  # noqa: F401
import torchtitan.models.llama3_ft  # noqa: F401

try:
    importlib.import_module("torchtitan.models.deepseek_v3")  # noqa: F401
except ImportError as e:
    warnings.warn(f"Skipping deepseek_v3: {e}")
