"""Pure-PyTorch hierarchical (tree) attention.

Public entrypoint: :func:`hierarchical_attention`.
"""

from treeattn_torch._tree_mha import hierarchical_attention

__all__ = ["hierarchical_attention"]
__version__ = "0.1.0"
