from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional
import logging

log = logging.getLogger(__name__)


@dataclass
class ModelContext:
    """Everything the loader needs to know about a repo, with zero task knowledge."""

    model_id: str
    config: Any = None
    is_diffusers: bool = False
    auto_map: dict = field(default_factory=dict)
    architectures: list = field(default_factory=list)
    model_type: Optional[str] = None
    is_encoder_decoder: bool = False
    has_image_processor: bool = False
    has_feature_extractor: bool = False
    has_tokenizer: bool = True

    # Convenience views ----------------------------------------------------
    @property
    def trust_remote_code(self) -> bool:
        # Custom code repos always need it; otherwise let the strategy decide.
        return bool(self.auto_map) or self.model_type in _REMOTE_MODEL_TYPES

    @property
    def primary_architecture(self) -> Optional[str]:
        return self.architectures[0] if self.architectures else None


# A *tiny* allowlist of model_types known to ship remote code (e.g. early MSA, mPLUG, etc.).
# This is NOT an architecture→loader map; it only flags trust_remote_code.
_REMOTE_MODEL_TYPES = set()  # populated lazily, see below


def _repo_files(model_id: str) -> set[str]:
    try:
        from huggingface_hub import HfApi
        return set(HfApi().list_repo_files(model_id))
    except Exception as e:  # offline / private / not found
        log.debug("Could not list repo files for %s: %s", model_id, e)
        return set()


def inspect(model_id: str, **config_kwargs) -> ModelContext:
    """
    Inspect a HF repo in priority order:
        1. Diffusers (model_index.json present)            -> short-circuit
        2. AutoConfig.auto_map                            -> custom code
        3. AutoConfig.architectures                       -> pattern-match suffix
        4. AutoConfig.model_type                          -> small fallback map
        5. AutoConfig.is_encoder_decoder                  -> structural fallback
    """
    try:
        from transformers import AutoConfig
    except ImportError as exc:
        raise RuntimeError(
            "Transformers is required only when inspecting a model on the server. "
            "Notebook cell generation does not require it."
        ) from exc

    files = _repo_files(model_id)
    if "model_index.json" in files:
        return ModelContext(model_id=model_id, is_diffusers=True, has_tokenizer=False)

    config = AutoConfig.from_pretrained(
        model_id, trust_remote_code=True, **config_kwargs
    )

    has_image_processor = (
        hasattr(config, "image_processor_type")
        or "preprocessor_config.json" in files
        or any("image" in a.lower() for a in (getattr(config, "architectures", []) or []))
    )
    has_feature_extractor = hasattr(config, "feature_extractor_type") or any(
        "speech" in a.lower() or "audio" in a.lower()
        for a in (getattr(config, "architectures", []) or [])
    )

    return ModelContext(
        model_id=model_id,
        config=config,
        is_diffusers=False,
        auto_map=getattr(config, "auto_map", {}) or {},
        architectures=getattr(config, "architectures", []) or [],
        model_type=getattr(config, "model_type", None),
        is_encoder_decoder=getattr(config, "is_encoder_decoder", False),
        has_image_processor=has_image_processor,
        has_feature_extractor=has_feature_extractor,
    )
    