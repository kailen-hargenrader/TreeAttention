"""CUDA-oriented tree attention package.

The first implementation slice provides a direct PyTorch autograd path for
stochastic non-causal tree attention while preserving the public
``hierarchical_attention`` entrypoint used by the benchmark harness.
"""

from treeattn_cuda._autograd import has_native_kernels, hierarchical_attention

__all__ = ["has_native_kernels", "hierarchical_attention"]
__version__ = "0.1.0"