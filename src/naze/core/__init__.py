"""Naze core: numerical foundation (Stage 1) + gradient check (Stage 2)."""

from naze.core.gradcheck import assert_close, numeric_grad
from naze.core.numeric import Array, DType, as_array, seeded_rng

__all__ = ["Array", "DType", "as_array", "seeded_rng", "numeric_grad", "assert_close"]
