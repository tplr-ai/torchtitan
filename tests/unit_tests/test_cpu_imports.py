# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import subprocess
import sys
import textwrap


def test_cpu_configuration_and_experts_without_triton():
    code = textwrap.dedent(
        """
        import sys
        import torch

        # Exercise a fresh TorchTitan import even on Linux hosts with Triton.
        sys.modules["triton"] = None
        from torchtitan.config import JobConfig
        from torchtitan.models.llama3 import Transformer
        from torchtitan.models.moe.moe import GroupedExperts
        from torchtitan.models.qwen3 import Qwen3Model

        JobConfig()
        model = GroupedExperts(
            dim=8, hidden_dim=16, num_experts=4, use_grouped_mm=False
        )
        model.init_weights(init_std=0.02)
        inputs = torch.randn(6, 8, requires_grad=True)
        counts = torch.tensor([2, 0, 1, 3], dtype=torch.int64)
        output = model(inputs, counts)
        assert output.shape == inputs.shape
        output.square().sum().backward()
        assert torch.isfinite(inputs.grad).all()
        assert torch.isfinite(model.w1.grad).all()
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
