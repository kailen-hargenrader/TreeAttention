"""CUDA-oriented tree attention package.

The first implementation slice provides a direct PyTorch autograd path for
stochastic non-causal tree attention while preserving the public
``hierarchical_attention`` entrypoint used by the benchmark harness.
"""

from treeattn_cuda._autograd import (
	get_runtime_stats,
	has_native_kernels,
	hierarchical_attention,
	reset_runtime_stats,
)

__all__ = [
	"get_runtime_stats",
	"has_native_kernels",
	"hierarchical_attention",
	"reset_runtime_stats",
]
__version__ = "0.1.0"