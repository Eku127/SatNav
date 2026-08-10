"""Uni-NaVid policy adapter for SatNav's common evaluator."""

from __future__ import annotations

import hashlib
from typing import Any, List, Mapping, Optional

from satnav.evaluation import EpisodeContext, PolicyStep

from baselines.vlm.uninavid.actions import (
    action_chunk_or_stop,
    build_prompt,
    episode_instruction,
)
from baselines.vlm.uninavid.checkpoint import load_strict_model


def _tensor_digest(tensor: Any) -> str:
    value = tensor.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(value).hexdigest()


def _rgb_identity(rgb: Any) -> Mapping[str, Any]:
    import numpy as np

    value = np.ascontiguousarray(rgb)
    return {
        "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
    }


def _agent_state(context: EpisodeContext) -> Mapping[str, Any]:
    state = context.agent_state
    return {
        "position": [float(value) for value in state.position],
        "rotation": float(state.rotation),
    }


class UniNaVidPolicyAdapter:
    """Own incremental frames, navigation cache, generation, and action queue."""

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        image_processor: Any,
        torch_module: Any,
        constants: Any,
        tokenizer_image_token: Any,
        keywords_stopping_criteria: Any,
        conversation_templates: Mapping[str, Any],
        separator_style: Any,
        device: Any,
        max_new_tokens: int = 1024,
        load_report: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.torch = torch_module
        self.constants = constants
        self.tokenizer_image_token = tokenizer_image_token
        self.keywords_stopping_criteria = keywords_stopping_criteria
        self.conversation_templates = conversation_templates
        self.separator_style = separator_style
        self.device = device
        self.max_new_tokens = int(max_new_tokens)
        self.load_report = dict(load_report or {})
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        self._context: Optional[EpisodeContext] = None
        self._instruction = ""
        self._pending_frames: List[Any] = []
        self._queue: List[int] = []
        self._query_id = 0
        self._closed = False

        def tokenized(text: str):
            return tokenizer(text, return_tensors="pt").input_ids[0][1:].to(device)

        self._tok_video_start = tokenized(constants.VIDEO_START_SPECIAL_TOKEN)
        self._tok_video_end = tokenized(constants.VIDEO_END_SPECIAL_TOKEN)
        self._tok_image_start = tokenized(constants.IMAGE_START_TOKEN)
        self._tok_image_end = tokenized(constants.IMAGE_END_TOKEN)
        self._tok_navigation = tokenized(constants.NAVIGATION_SPECIAL_TOKEN)
        self._tok_image_separator = tokenized(constants.IAMGE_SEPARATOR)

    @classmethod
    def from_pretrained(
        cls,
        model_path,
        *,
        uninavid_repo,
        eva_path,
        processor_path,
        device: str = "cuda:0",
        max_new_tokens: int = 1024,
        flash_attention: bool = True,
    ) -> "UniNaVidPolicyAdapter":
        tokenizer, model, image_processor, load_report = load_strict_model(
            model_path,
            uninavid_repo=uninavid_repo,
            eva_path=eva_path,
            processor_path=processor_path,
            device=device,
            flash_attention=flash_attention,
        )
        import torch
        import uninavid.constants as constants
        from uninavid.conversation import SeparatorStyle, conv_templates
        from uninavid.mm_utils import KeywordsStoppingCriteria, tokenizer_image_token

        return cls(
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            torch_module=torch,
            constants=constants,
            tokenizer_image_token=tokenizer_image_token,
            keywords_stopping_criteria=KeywordsStoppingCriteria,
            conversation_templates=conv_templates,
            separator_style=SeparatorStyle,
            device=torch.device(device),
            max_new_tokens=max_new_tokens,
            load_report=load_report,
        )

    def reset(self, context: EpisodeContext) -> None:
        if self._closed:
            raise RuntimeError("UniNaVidPolicyAdapter is closed")
        self._context = context
        self._instruction = episode_instruction(context.episode)
        self._pending_frames.clear()
        self._queue.clear()
        self._query_id = 0
        self.torch.manual_seed(context.seed)
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed_all(context.seed)
        self.model.config.run_type = "eval"
        self.model.get_model().initialize_online_inference_nav_feat_cache()
        self.model.get_model().new_frames = 0
        self.model.eval()

    def _cache_state(self) -> Mapping[str, Any]:
        core = self.model.get_model()
        result = {"new_frames": int(getattr(core, "new_frames", -1))}
        for name in ("feat_cache", "long_feat_cache"):
            value = getattr(core, name, None)
            result[name] = None if value is None else list(value.shape)
        return result

    def _process_images(self, frames: List[Any]) -> List[Any]:
        import numpy as np

        self.model.get_model().new_frames = len(frames)
        video = (
            self.image_processor.preprocess(
                np.asarray(frames), return_tensors="pt"
            )["pixel_values"]
            .half()
            .to(self.device)
        )
        return [video]

    def _predict(self, frames: List[Any]) -> tuple[str, Mapping[str, Any]]:
        c = self.constants
        prompt = build_prompt(self._instruction, c.NAVIGATION_IDENTIFIER, c.DEFAULT_IMAGE_TOKEN)
        question = prompt.replace(c.DEFAULT_IMAGE_TOKEN, "").replace("\n", "")
        if self.model.config.mm_use_im_start_end:
            query = (
                c.DEFAULT_IM_START_TOKEN
                + c.DEFAULT_IMAGE_TOKEN
                + c.DEFAULT_IM_END_TOKEN
                + "\n"
                + prompt.replace(c.DEFAULT_IMAGE_TOKEN, "")
            )
        else:
            query = c.DEFAULT_IMAGE_TOKEN + "\n" + prompt.replace(c.DEFAULT_IMAGE_TOKEN, "")
        conversation = self.conversation_templates["vicuna_v1"].copy()
        conversation.append_message(conversation.roles[0], query)
        conversation.append_message(conversation.roles[1], None)
        full_prompt = conversation.get_prompt()
        tokens = self.tokenizer_image_token(
            full_prompt,
            self.tokenizer,
            c.IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).to(self.device)
        indices = self.torch.where(tokens == c.IMAGE_TOKEN_INDEX)[0]
        parts = []
        while indices.numel() > 0:
            index = indices[0].item()
            parts.extend(
                (
                    tokens[:index],
                    self._tok_video_start,
                    self._tok_image_separator,
                    tokens[index : index + 1],
                    self._tok_video_end,
                    self._tok_image_start,
                    self._tok_image_end,
                    self._tok_navigation,
                )
            )
            tokens = tokens[index + 1 :]
            indices = self.torch.where(tokens == c.IMAGE_TOKEN_INDEX)[0]
        if tokens.numel():
            parts.append(tokens)
        input_ids = self.torch.cat(parts).unsqueeze(0)
        stop_string = (
            conversation.sep
            if conversation.sep_style != self.separator_style.TWO
            else conversation.sep2
        )
        stopping = self.keywords_stopping_criteria(
            [stop_string], self.tokenizer, input_ids
        )
        cache_before = self._cache_state()
        images = self._process_images(frames)
        self.model.update_prompt([[question]])
        with self.torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images,
                do_sample=False,
                temperature=0.0,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping],
            )
        suffix = (
            output_ids[:, input_ids.shape[1] :]
            if output_ids.shape[1] >= input_ids.shape[1]
            else output_ids
        )
        output = self.tokenizer.batch_decode(suffix, skip_special_tokens=True)[0].strip()
        if stop_string and output.endswith(stop_string):
            output = output[: -len(stop_string)]
        diagnostics = {
            "query_id": self._query_id,
            "prompt": prompt,
            "rendered_prompt": full_prompt,
            "input_ids_sha256": _tensor_digest(input_ids),
            "input_token_count": int(input_ids.numel()),
            "generated_token_ids": suffix.detach().cpu().tolist()[0],
            "generated_ids_sha256": _tensor_digest(suffix),
            "new_frame_count": len(frames),
            "new_frame_identities": [_rgb_identity(frame) for frame in frames],
            "cache_before": cache_before,
            "cache_after": self._cache_state(),
        }
        self._query_id += 1
        return output.strip(), diagnostics

    def act(self, observation: Mapping[str, Any]) -> PolicyStep:
        if self._context is None:
            raise RuntimeError("reset() must be called before act()")
        self._pending_frames.append(observation["rgb"])
        diagnostics = {
            "observation": _rgb_identity(observation["rgb"]),
            "agent_state_before": _agent_state(self._context),
        }
        if self._queue:
            action = self._queue.pop(0)
            diagnostics.update(source="queued", queue_remaining=len(self._queue))
        else:
            frames = self._pending_frames
            self._pending_frames = []
            output, query = self._predict(frames)
            actions, fallback = action_chunk_or_stop(output)
            action = actions.pop(0)
            self._queue.extend(actions)
            diagnostics.update(
                source="generated",
                generated_text=output,
                parsed_queue=[action, *actions],
                fallback_stop=fallback,
                queue_remaining=len(self._queue),
                query=query,
            )
        return PolicyStep(action=action, info=diagnostics)

    def close(self) -> None:
        self._context = None
        self._pending_frames.clear()
        self._queue.clear()
        self._closed = True
