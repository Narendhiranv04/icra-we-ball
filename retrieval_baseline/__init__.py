"""Open-vocabulary retrieval baseline: no language model anywhere.

Roles are a fixed, non-privileged task template; every role is filled by CLIP
image-text similarity between the role's *function phrase* and a crop of the
candidate taken from the raw (unannotated) camera frame.
"""

from .retrieval import CLIPRetriever, RetrievalScores, load_clip

__all__ = ["CLIPRetriever", "RetrievalScores", "load_clip"]
