"""Systematics preprocessing helpers (TREx replacement)."""

from .hygiene import apply_negative_bins_policy as apply_negative_bins_policy
from .hygiene import NegativeBinsPolicy as NegativeBinsPolicy
from .pipeline import NegativeBinsHygieneStep as NegativeBinsHygieneStep
from .pipeline import PreprocessPipeline as PreprocessPipeline
from .pipeline import PruneSystematicsStep as PruneSystematicsStep
from .pipeline import SmoothHistoSysStep as SmoothHistoSysStep
from .pipeline import SymmetrizeHistoSysStep as SymmetrizeHistoSysStep
from .prune import PruneDecision as PruneDecision
from .prune import PruneMethod as PruneMethod
from .prune import should_prune_histosys_overall as should_prune_histosys_overall
from .prune import should_prune_histosys_shape as should_prune_histosys_shape
from .prune import should_prune_normsys as should_prune_normsys
from .smooth import SmoothMethod as SmoothMethod
from .smooth import SmoothResult as SmoothResult
from .smooth import smooth_variation as smooth_variation
from .symmetrize import symmetrize_shapes as symmetrize_shapes

__all__ = [
    "NegativeBinsHygieneStep",
    "NegativeBinsPolicy",
    "PreprocessPipeline",
    "PruneDecision",
    "PruneMethod",
    "PruneSystematicsStep",
    "SmoothHistoSysStep",
    "SmoothMethod",
    "SmoothResult",
    "SymmetrizeHistoSysStep",
    "apply_negative_bins_policy",
    "should_prune_histosys_overall",
    "should_prune_histosys_shape",
    "should_prune_normsys",
    "smooth_variation",
    "symmetrize_shapes",
]
