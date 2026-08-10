"""NaVILA policy adapter for SatNav's framework-independent evaluator."""

from __future__ import annotations

from typing import Any, List, Mapping, Optional

from satnav.evaluation import EpisodeContext, PolicyStep

from baselines.vlm.navila.actions import (
    build_prompt,
    episode_instruction,
    normalize_action_format,
    parse_action_text,
    parse_action_queue,
    sample_and_pad_images,
)
from baselines.vlm.navila.bootstrap import bootstrap_navila


class SafeKeywordsStoppingCriteria:
    """Build the pinned stopping criterion without matching prompt-only tokens."""

    @staticmethod
    def build(criteria_class: Any, keywords: List[str], tokenizer: Any, input_ids: Any):
        class _SafeCriteria(criteria_class):
            def call_for_batch(self, output_ids, scores, **kwargs):
                generated = output_ids.shape[1] - self.start_len
                if generated <= 0:
                    return False
                offset = min(generated, self.max_keyword_len)
                self.keyword_ids = [
                    value.to(output_ids.device) for value in self.keyword_ids
                ]
                for keyword_id in self.keyword_ids:
                    if (
                        keyword_id.shape[0] <= generated
                        and (output_ids[0, -keyword_id.shape[0] :] == keyword_id).all()
                    ):
                        return True
                recent = output_ids[:, -offset:]
                decoded = self.tokenizer.batch_decode(recent, skip_special_tokens=True)[
                    0
                ]
                return any(keyword in decoded for keyword in self.keywords)

        return _SafeCriteria(keywords, tokenizer, input_ids)


def _generated_suffix(output_ids: Any, input_length: int) -> Any:
    """Accept both full-sequence and generated-suffix-only model returns."""

    if output_ids.shape[1] >= input_length:
        return output_ids[:, input_length:]
    return output_ids


class NaVILAPolicyAdapter:
    """Own NaVILA image history, generation, and primitive action queue."""

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        image_processor: Any,
        torch_module: Any,
        process_images: Any,
        tokenizer_image_token: Any,
        keywords_stopping_criteria: Any,
        image_token_index: int,
        conversation_templates: Mapping[str, Any],
        separator_style: Any,
        device: Any,
        dtype: Any,
        num_frames: int = 8,
        max_new_tokens: int = 32,
        action_format: str = "compact",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.torch = torch_module
        self.process_images = process_images
        self.tokenizer_image_token = tokenizer_image_token
        self.keywords_stopping_criteria = keywords_stopping_criteria
        self.image_token_index = int(image_token_index)
        self.conversation_templates = conversation_templates
        self.separator_style = separator_style
        self.device = device
        self.dtype = dtype
        self.num_frames = int(num_frames)
        self.max_new_tokens = int(max_new_tokens)
        self.action_format = normalize_action_format(action_format)
        if self.num_frames < 2 or self.max_new_tokens <= 0:
            raise ValueError("num_frames >= 2 and max_new_tokens > 0 are required")
        self._context: Optional[EpisodeContext] = None
        self._instruction = ""
        self._history: List[Any] = []
        self._queue: List[int] = []
        self._closed = False

    @classmethod
    def from_pretrained(
        cls,
        model_path,
        *,
        navila_repo=None,
        require_pinned_revision: bool = True,
        device: str = "cuda:0",
        dtype: str = "float16",
        num_frames: int = 8,
        max_new_tokens: int = 32,
        action_format: str = "compact",
    ) -> "NaVILAPolicyAdapter":
        """Load the pinned external model only after evaluator setup."""

        bootstrap_navila(navila_repo, require_pinned_revision=require_pinned_revision)
        import torch

        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.conversation import SeparatorStyle, conv_templates
        from llava.mm_utils import (
            KeywordsStoppingCriteria,
            process_images,
            tokenizer_image_token,
        )
        from llava.model.builder import load_pretrained_model

        if dtype != "float16":
            raise ValueError(
                "the pinned NaVILA loader supports dtype=float16 for evaluation"
            )
        torch_device = torch.device(device)
        tokenizer, model, image_processor, _ = load_pretrained_model(
            str(model_path),
            str(model_path).rstrip("/").split("/")[-1],
            model_base=None,
            device_map={"": torch_device},
            device=str(torch_device),
        )
        model.to(torch_device)
        model.eval()
        return cls(
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            torch_module=torch,
            process_images=process_images,
            tokenizer_image_token=tokenizer_image_token,
            keywords_stopping_criteria=KeywordsStoppingCriteria,
            image_token_index=IMAGE_TOKEN_INDEX,
            conversation_templates=conv_templates,
            separator_style=SeparatorStyle,
            device=torch_device,
            dtype=torch.float16,
            num_frames=num_frames,
            max_new_tokens=max_new_tokens,
            action_format=action_format,
        )

    def reset(self, context: EpisodeContext) -> None:
        if self._closed:
            raise RuntimeError("NaVILAPolicyAdapter is closed")
        self._context = context
        self._instruction = episode_instruction(context.episode)
        self._history.clear()
        self._queue.clear()
        self.torch.manual_seed(context.seed)
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed_all(context.seed)
        self.model.eval()

    def _predict(self, frames: List[Any]) -> str:
        question = build_prompt(
            self._instruction,
            history_count=self.num_frames - 1,
            action_format=self.action_format,
        )
        conversation = self.conversation_templates["llama_3"].copy()
        conversation.append_message(conversation.roles[0], question)
        conversation.append_message(conversation.roles[1], None)
        prompt = conversation.get_prompt()
        image_tensor = self.process_images(
            frames, self.image_processor, self.model.config
        ).to(self.device, dtype=self.dtype)
        input_ids = (
            self.tokenizer_image_token(
                prompt,
                self.tokenizer,
                self.image_token_index,
                return_tensors="pt",
            )
            .unsqueeze(0)
            .to(self.device)
        )
        stop_string = (
            conversation.sep
            if conversation.sep_style != self.separator_style.TWO
            else conversation.sep2
        )
        stopping = SafeKeywordsStoppingCriteria.build(
            self.keywords_stopping_criteria,
            [stop_string],
            self.tokenizer,
            input_ids,
        )
        with self.torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping],
                pad_token_id=self.tokenizer.eos_token_id,
            )
        suffix = _generated_suffix(output_ids, input_ids.shape[1])
        output = self.tokenizer.batch_decode(suffix, skip_special_tokens=True)[
            0
        ].strip()
        if stop_string and output.endswith(stop_string):
            output = output[: -len(stop_string)]
        return output.strip()

    def act(self, observation: Mapping[str, Any]) -> PolicyStep:
        if self._context is None:
            raise RuntimeError("reset() must be called before act()")
        from PIL import Image

        current = Image.fromarray(observation["rgb"]).convert("RGB")
        generated = None
        if self._queue:
            action = self._queue.pop(0)
            source = "queued"
        else:
            sampled = sample_and_pad_images(
                self._history + [current], num_frames=self.num_frames
            )
            generated = self._predict(sampled)
            parse_fallback = parse_action_text(generated) is None
            action, queued = parse_action_queue(generated)
            self._queue.extend(queued)
            source = "generated"
        self._history.append(current)
        info = {
            "source": source,
            "queue_remaining": len(self._queue),
            "action_format": self.action_format,
        }
        if generated is not None:
            info["generated_text"] = generated
            info["parse_fallback"] = parse_fallback
        return PolicyStep(action=action, info=info)

    def close(self) -> None:
        self._context = None
        self._history.clear()
        self._queue.clear()
        self._closed = True
