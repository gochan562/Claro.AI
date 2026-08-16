from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Optional

from .model_inspector import ModelContext


@dataclass
class LoaderPlan:
    auto_model_class: str            # e.g. "AutoModelForCausalLM"
    preprocessor_classes: list[str]  # e.g. ["AutoTokenizer"] or ["AutoProcessor","AutoImageProcessor"]
    trust_remote_code: bool
    extra_imports: list[str]         # e.g. ["from diffusers import DiffusionPipeline"]
    loader_call: str                 # the actual from_pretrained expression
    preprocessor_call: str
    model_var: str = "model"
    preprocessor_var: str = "tokenizer"


class LoaderStrategy:
    @classmethod
    def matches(cls, ctx: ModelContext) -> bool: ...
    @classmethod
    def build(cls, ctx: ModelContext) -> LoaderPlan: ...


# ---------------------------------------------------------------- Diffusers
class DiffusersStrategy(LoaderStrategy):
    @classmethod
    def matches(cls, ctx: ModelContext) -> bool:
        return ctx.is_diffusers

    @classmethod
    def build(cls, ctx: ModelContext) -> LoaderPlan:
        return LoaderPlan(
            auto_model_class="DiffusionPipeline",
            preprocessor_classes=[],
            trust_remote_code=False,
            extra_imports=["from diffusers import DiffusionPipeline"],
            loader_call=f"DiffusionPipeline.from_pretrained('{ctx.model_id}', dtype=torch.float16)",
            preprocessor_call="",
            model_var="pipe",
            preprocessor_var="",
        )


# ---------------------------------------------------------------- AutoMap
# Repo ships its own modeling_*.py. Trust it and use AutoModelFor... directly;
# the auto_map usually names the base class, so we still pattern-match the
# architecture suffix to pick the right AutoModel head.
class AutoMapStrategy(LoaderStrategy):
    @classmethod
    def matches(cls, ctx: ModelContext) -> bool:
        return bool(ctx.auto_map) and not ctx.is_diffusers

    @classmethod
    def build(cls, ctx: ModelContext) -> LoaderPlan:
        # Defer to ArchitecturePatternStrategy for class selection, but force trust_remote_code.
        plan = ArchitecturePatternStrategy.build(ctx)
        plan.trust_remote_code = True
        return plan


# ---------------------------------------------------------------- Architecture suffix
# Architecture names expose the task head/interface.  These are interface
# suffixes, not a model-family allowlist, so a new family with a conventional
# Transformers architecture name works without a code change.
_ARCH_SUFFIX_TO_AUTOCLASS = (
    (r"ForQuestionAnswering$", "AutoModelForQuestionAnswering", ["AutoTokenizer"]),
    (r"ForSequenceClassification$", "AutoModelForSequenceClassification", ["AutoTokenizer"]),
    (r"ForTokenClassification$", "AutoModelForTokenClassification", ["AutoTokenizer"]),
    (r"ForMultipleChoice$", "AutoModelForMultipleChoice", ["AutoTokenizer"]),
    (r"ForMaskedLM$", "AutoModelForMaskedLM", ["AutoTokenizer"]),
    (r"ForCausalLM$", "AutoModelForCausalLM", ["AutoTokenizer"]),
    (r"ForImageTextToText$", "AutoModelForImageTextToText", ["AutoProcessor"]),
    (r"ForVision2Seq$", "AutoModelForVision2Seq", ["AutoProcessor"]),
    (r"ForImageClassification$", "AutoModelForImageClassification", ["AutoImageProcessor"]),
    (r"ForObjectDetection$", "AutoModelForObjectDetection", ["AutoImageProcessor"]),
    (r"ForSemanticSegmentation$", "AutoModelForSemanticSegmentation", ["AutoImageProcessor"]),
    (r"ForDepthEstimation$", "AutoModelForDepthEstimation", ["AutoImageProcessor"]),
    (r"ForCTC$", "AutoModelForCTC", ["AutoProcessor"]),
    (r"ForAudioClassification$", "AutoModelForAudioClassification", ["AutoFeatureExtractor"]),
    (r"ForSpeechSeq2Seq$", "AutoModelForSpeechSeq2Seq", ["AutoProcessor"]),
)


class ArchitecturePatternStrategy(LoaderStrategy):
    @classmethod
    def matches(cls, ctx: ModelContext) -> bool:
        return any(
            re.search(suffix, arch)
            for arch in ctx.architectures
            for suffix, _, _ in _ARCH_SUFFIX_TO_AUTOCLASS
        )

    @classmethod
    def build(cls, ctx: ModelContext) -> LoaderPlan:
        autoclass, preproc = None, ["AutoTokenizer"]
        for arch in ctx.architectures:
            for suffix, ac, pp in _ARCH_SUFFIX_TO_AUTOCLASS:
                if re.search(suffix, arch):
                    autoclass, preproc = ac, pp
                    break
            if autoclass:
                break

        # "*ForConditionalGeneration" is ambiguous: T5/BART/BART→Seq2Seq, but
        # Llava/Donut→Vision2Seq. Disambiguate structurally.
        if autoclass is None:
            if ctx.has_image_processor:
                autoclass, preproc = "AutoModelForVision2Seq", ["AutoProcessor"]
            elif ctx.is_encoder_decoder:
                autoclass, preproc = "AutoModelForSeq2SeqLM", ["AutoTokenizer"]
            else:
                autoclass, preproc = "AutoModelForCausalLM", ["AutoTokenizer"]

        return _assemble_plan(ctx, autoclass, preproc)


# ---------------------------------------------------------------- model_type fallback
# Only used when architectures is empty.  model_type can identify a small
# number of multimodal families, but it cannot identify a BERT QA head by
# itself, so generic text models are resolved structurally instead of by task.
_MODEL_TYPE_TO_AUTOCLASS = {
    "donut": ("AutoModelForVision2Seq", ["AutoProcessor"]),
    "vision-encoder-decoder": ("AutoModelForVision2Seq", ["AutoProcessor"]),
}


class ModelTypeStrategy(LoaderStrategy):
    @classmethod
    def matches(cls, ctx: ModelContext) -> bool:
        return ctx.model_type in _MODEL_TYPE_TO_AUTOCLASS

    @classmethod
    def build(cls, ctx: ModelContext) -> LoaderPlan:
        ac, pp = _MODEL_TYPE_TO_AUTOCLASS[ctx.model_type]
        return _assemble_plan(ctx, ac, pp)


# ---------------------------------------------------------------- Structural fallback
class StructuralFallbackStrategy(LoaderStrategy):
    @classmethod
    def matches(cls, ctx: ModelContext) -> bool: return True

    @classmethod
    def build(cls, ctx: ModelContext) -> LoaderPlan:
        if ctx.has_image_processor:
            ac, pp = "AutoModelForVision2Seq", ["AutoProcessor"]
        elif ctx.is_encoder_decoder:
            ac, pp = "AutoModelForSeq2SeqLM", ["AutoTokenizer"]
        else:
            ac, pp = "AutoModelForCausalLM", ["AutoTokenizer"]
        return _assemble_plan(ctx, ac, pp)


# ---------------------------------------------------------------- Helpers
def _assemble_plan(ctx: ModelContext, autoclass: str, preproc: list[str]) -> LoaderPlan:
    trc = ctx.trust_remote_code
    trc_kw = ", trust_remote_code=True" if trc else ""

    # If the model is vision-language, prefer AutoProcessor (covers LightOnOCR, Donut, Llava, etc.)
    if ctx.has_image_processor and "AutoTokenizer" in preproc:
        preproc = ["AutoProcessor"]

    pp_var = (
        "processor"
        if any(cls in preproc for cls in ("AutoProcessor", "AutoImageProcessor"))
        else "feature_extractor"
        if "AutoFeatureExtractor" in preproc
        else "tokenizer"
    )
    pp_calls = [
        f"{cls}.from_pretrained('{ctx.model_id}'{trc_kw})" for cls in preproc
    ]
    return LoaderPlan(
        auto_model_class=autoclass,
        preprocessor_classes=preproc,
        trust_remote_code=trc,
        extra_imports=[],
        loader_call=f"{autoclass}.from_pretrained('{ctx.model_id}'{trc_kw})",
        preprocessor_call="; ".join(
            f"{pp_var} = {c}" for c in pp_calls
        ) or f"{pp_var} = None",
        model_var="model",
        preprocessor_var=pp_var,
    )


# ---------------------------------------------------------------- Registry
REGISTRY: list[type[LoaderStrategy]] = [
    DiffusersStrategy,
    AutoMapStrategy,
    ArchitecturePatternStrategy,
    ModelTypeStrategy,
    StructuralFallbackStrategy,
]


def resolve_loader(ctx: ModelContext) -> LoaderPlan:
    for strat in REGISTRY:
        if strat.matches(ctx):
            return strat.build(ctx)
    raise RuntimeError(f"No loader strategy matched for {ctx.model_id}")