"""OpenFly policy adapter for SatNav's framework-independent evaluator."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from satnav.evaluation import EpisodeContext, PolicyStep

from baselines.vlm.openfly.actions import (
    AUTO,
    COMPACT,
    ORIGINAL,
    ORIGINAL_UNNORM_KEY,
    convert_original_action_vector_to_action,
    parse_action_text,
    resolve_action_format,
    validate_tokenizer_model_contract,
)
from baselines.vlm.openfly.artifacts import openfly_model_identity
from baselines.vlm.openfly.bootstrap import register_openfly_classes
from baselines.vlm.openfly.backends.base import reject_non_strict_loading
from baselines.vlm.openfly.prompting import build_openfly_prompt


def episode_instruction(episode: Any) -> str:
    value = getattr(episode, "instruction", None)
    if isinstance(value, Mapping):
        value = value.get("instruction_text", value.get("text"))
    elif value is not None and not isinstance(value, str):
        value = getattr(value, "instruction_text", getattr(value, "text", value))
    if value is None:
        value = getattr(episode, "instructions", None)
        if isinstance(value, Sequence) and not isinstance(value, str) and value:
            value = value[0]
    rendered = " ".join(str(value or "").split())
    if not rendered:
        raise ValueError("SatNav episode has no instruction text")
    return rendered


def _torch_dtype(torch_module: Any, name: str):
    mapping = {
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "float32": torch_module.float32,
    }
    try:
        return mapping[str(name)]
    except KeyError as error:
        raise ValueError(f"unsupported OpenFly inference dtype: {name!r}") from error


class OpenFlyPolicyAdapter:
    """Own model generation plus the exact three-frame/action-history state."""

    def __init__(
        self,
        *,
        model: Any,
        processor: Any,
        torch_module: Any,
        device: Any,
        dtype: Any,
        action_format: str,
        max_new_tokens: int = 8,
        action_history_limit: int = 16,
        unnorm_key: str = ORIGINAL_UNNORM_KEY,
    ) -> None:
        self.model = model
        self.processor = processor
        self.torch = torch_module
        self.device = device
        self.dtype = dtype
        self.action_format = resolve_action_format(action_format)
        self.max_new_tokens = int(max_new_tokens)
        self.action_history_limit = int(action_history_limit)
        self.unnorm_key = str(unnorm_key)
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.action_history_limit < 0:
            raise ValueError("action_history_limit must be non-negative")
        if self.action_format == ORIGINAL:
            if not hasattr(self.model, "get_action_stats"):
                raise TypeError("OpenFly original checkpoint lacks action statistics")
            stats = self.model.get_action_stats(self.unnorm_key)
            if len(stats["q01"]) != 8:
                raise ValueError("OpenFly original action statistics must have 8 dimensions")
        self._context: Optional[EpisodeContext] = None
        self._instruction = ""
        self._frames: list[Any] = []
        self._actions: list[int] = []
        self._closed = False

    @classmethod
    def from_pretrained(
        cls,
        model_path: Path,
        *,
        device: str = "cuda:0",
        dtype: str = "float16",
        action_format: str = AUTO,
        max_new_tokens: int = 8,
        action_history_limit: int = 16,
        unnorm_key: str = ORIGINAL_UNNORM_KEY,
    ) -> "OpenFlyPolicyAdapter":
        model_path = Path(model_path).expanduser().resolve()
        identity = openfly_model_identity(model_path)
        declared = str(identity["facts"]["action_format"])
        resolved_format = resolve_action_format(action_format, declared)
        if resolved_format != declared:
            raise ValueError(
                "requested action format contradicts checkpoint metadata: "
                f"requested={resolved_format}, checkpoint={declared}"
            )
        register_openfly_classes()
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        torch_device = torch.device(device)
        torch_dtype = _torch_dtype(torch, dtype)
        processor = AutoProcessor.from_pretrained(
            model_path, local_files_only=True
        )
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token = (
                processor.tokenizer.eos_token or processor.tokenizer.unk_token
            )
        if processor.tokenizer.pad_token_id is None:
            raise ValueError("OpenFly tokenizer has no usable padding token")
        model, loading_info = AutoModelForVision2Seq.from_pretrained(
            model_path,
            local_files_only=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch_dtype,
            output_loading_info=True,
        )
        reject_non_strict_loading(loading_info, "OpenFly evaluation checkpoint")
        validate_tokenizer_model_contract(model, processor.tokenizer, resolved_format)
        model.to(torch_device)
        model.eval()
        return cls(
            model=model,
            processor=processor,
            torch_module=torch,
            device=torch_device,
            dtype=torch_dtype,
            action_format=resolved_format,
            max_new_tokens=max_new_tokens,
            action_history_limit=action_history_limit,
            unnorm_key=unnorm_key,
        )

    def reset(self, context: EpisodeContext) -> None:
        if self._closed:
            raise RuntimeError("OpenFlyPolicyAdapter is closed")
        self._context = context
        self._instruction = episode_instruction(context.episode)
        self._frames.clear()
        self._actions.clear()
        self.torch.manual_seed(context.seed)
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed_all(context.seed)
        self.model.eval()

    def _three_frames(self, current: Any) -> list[Any]:
        history_first = self._frames[-1] if self._frames else current
        history_second = self._frames[-2] if len(self._frames) >= 2 else history_first
        return [current, history_first, history_second]

    def _inputs(self, current: Any) -> Mapping[str, Any]:
        prompt = build_openfly_prompt(
            self._instruction,
            self._actions,
            self.action_history_limit,
        )
        encoded = self.processor(
            text=prompt,
            images=self._three_frames(current),
            return_tensors="pt",
        )
        return {
            name: value.to(self.device)
            for name, value in encoded.items()
            if name in {"input_ids", "attention_mask", "pixel_values"}
        }

    def _compact_action(self, inputs: Mapping[str, Any]) -> tuple[int, Mapping[str, Any]]:
        input_ids = inputs["input_ids"]
        with self.torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )
        if (
            output_ids.shape[1] >= input_ids.shape[1]
            and self.torch.equal(output_ids[:, : input_ids.shape[1]], input_ids)
        ):
            generated_ids = output_ids[:, input_ids.shape[1] :]
        else:
            generated_ids = output_ids
        generated = self.processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()
        action = parse_action_text(generated)
        if action is None:
            raise ValueError(
                f"OpenFly compact output contains no canonical action: {generated!r}"
            )
        return action, {"generated_text": generated}

    def _original_action(self, inputs: Mapping[str, Any]) -> tuple[int, Mapping[str, Any]]:
        if not hasattr(self.model, "predict_action"):
            raise TypeError("OpenFly original checkpoint lacks predict_action()")
        with self.torch.inference_mode():
            raw = self.model.predict_action(
                **inputs,
                unnorm_key=self.unnorm_key,
                do_sample=False,
                use_cache=True,
            )
        action, rounded, distance = convert_original_action_vector_to_action(raw)
        return action, {
            "raw_action_vector": [float(value) for value in raw],
            "rounded_action_vector": [int(value) for value in rounded],
            "template_distance": float(distance),
        }

    def act(self, observation: Mapping[str, Any]) -> PolicyStep:
        if self._context is None:
            raise RuntimeError("reset() must be called before act()")
        if "rgb" not in observation:
            raise KeyError("OpenFly requires the SatNav rgb observation")
        from PIL import Image

        current = Image.fromarray(observation["rgb"]).convert("RGB")
        inputs = self._inputs(current)
        if self.action_format == COMPACT:
            action, details = self._compact_action(inputs)
        elif self.action_format == ORIGINAL:
            action, details = self._original_action(inputs)
        else:  # pragma: no cover - resolved during construction
            raise AssertionError(self.action_format)
        self._frames.append(current.copy())
        self._actions.append(action)
        return PolicyStep(
            action=action,
            info={
                "action_format": self.action_format,
                "action_history_length": len(self._actions),
                **details,
            },
        )

    def close(self) -> None:
        self._context = None
        self._frames.clear()
        self._actions.clear()
        self._closed = True
