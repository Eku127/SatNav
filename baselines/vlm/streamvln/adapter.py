"""StreamVLN model adapter for the framework-independent SatNav evaluator."""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any, List, Mapping, Optional

from satnav.evaluation import EpisodeContext, PolicyStep

from baselines.vlm.streamvln.actions import parse_action_symbols
from baselines.vlm.streamvln.bootstrap import bootstrap_streamvln
from baselines.vlm.streamvln.history import StreamingHistory


DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_MEMORY_TOKEN = "<memory>"
DEFAULT_VIDEO_TOKEN = "<video>"


def episode_instruction(episode: Any) -> str:
    """Extract instruction text from all SatNav v0.1-compatible shapes."""

    instruction = getattr(episode, "instruction", "")
    if isinstance(instruction, Mapping):
        return str(
            instruction.get("text", instruction.get("instruction_text", ""))
        )
    if hasattr(instruction, "text"):
        return str(instruction.text)
    if hasattr(instruction, "instruction_text"):
        return str(instruction.instruction_text)
    return str(instruction)


class StreamVLNPolicyAdapter:
    """Own StreamVLN streaming state behind SatNav's ``PolicyAdapter``.

    The generic evaluator requests exactly one primitive action at a time.
    This adapter retains a generated symbolic action chunk, Qwen output IDs,
    KV cache, all RGB observations, and fixed-window time IDs between calls.
    At every ``num_frames`` primitive actions only model/KV window state is
    reset; queued actions and global observation history are intentionally kept,
    matching the pinned adapter behavior.
    """

    PROMPT = (
        "You are an autonomous navigation assistant. "
        "Your task is to <instruction>. "
        "Devise an action sequence to follow the instruction using the four actions: "
        "TURN LEFT (←) or TURN RIGHT (→) by 15 degrees, "
        "MOVE FORWARD (↑) by 10 meters, or STOP."
    )
    CONJUNCTIONS = (
        "you can see ",
        "in front of you is ",
        "there is ",
        "you can spot ",
        "you are toward the ",
        "ahead of you is ",
        "in your sight is ",
    )

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        image_processor: Any,
        torch_module: Any,
        dict_to_device: Any,
        image_token_index: int,
        memory_token_index: int,
        device: Any,
        image_dtype: Any = None,
        num_frames: int = 32,
        num_history: Optional[int] = 8,
        num_future_steps: int = 4,
        max_new_tokens: int = 10000,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.torch = torch_module
        self.dict_to_device = dict_to_device
        self.image_token_index = int(image_token_index)
        self.memory_token_index = int(memory_token_index)
        self.device = device
        self.image_dtype = image_dtype or torch_module.bfloat16
        self.num_frames = int(num_frames)
        self.num_history = None if num_history is None else int(num_history)
        self.num_future_steps = int(num_future_steps)
        self.max_new_tokens = int(max_new_tokens)
        if self.num_frames <= 0 or self.num_future_steps <= 0:
            raise ValueError("num_frames and num_future_steps must be positive")
        if self.num_history is not None and self.num_history <= 0:
            raise ValueError("num_history must be positive or None")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")

        self.history = StreamingHistory(self.num_frames)
        self._context: Optional[EpisodeContext] = None
        self._instruction = ""
        self._action_queue: List[int] = []
        self._output_ids: Any = None
        self._past_key_values: Any = None
        self._model_world_size: Optional[int] = None
        self._rng = random.Random(0)
        self._closed = False

    @classmethod
    def from_pretrained(
        cls,
        model_path: Path,
        *,
        tokenizer_path: Optional[Path] = None,
        vision_tower: Optional[str] = None,
        streamvln_repo: Optional[Path] = None,
        require_pinned_revision: bool = True,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        attention_implementation: str = "flash_attention_2",
        num_frames: int = 32,
        num_history: Optional[int] = 8,
        num_future_steps: int = 4,
        max_new_tokens: int = 10000,
    ) -> "StreamVLNPolicyAdapter":
        """Load the pinned external model without importing it at module import."""

        bootstrap_streamvln(
            streamvln_repo, require_pinned_revision=require_pinned_revision
        )
        import torch
        import transformers

        from streamvln.model.stream_video_vln import StreamVLNForCausalLM
        from streamvln.utils.utils import (
            IMAGE_TOKEN_INDEX,
            MEMORY_TOKEN_INDEX,
            dict_to_cuda,
        )

        model_path = Path(model_path).expanduser().resolve()
        tokenizer_path = Path(tokenizer_path or model_path).expanduser().resolve()
        if not (model_path / "config.json").is_file():
            raise FileNotFoundError(f"StreamVLN config not found: {model_path / 'config.json'}")
        if not (tokenizer_path / "tokenizer_config.json").is_file():
            raise FileNotFoundError(
                f"Tokenizer config not found: {tokenizer_path / 'tokenizer_config.json'}"
            )

        dtype_value = getattr(torch, dtype, None)
        if dtype_value is None:
            raise ValueError(f"unsupported torch dtype: {dtype}")
        torch_device = torch.device(device)

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            str(tokenizer_path),
            model_max_length=32768,
            padding_side="right",
            local_files_only=True,
        )
        config = transformers.AutoConfig.from_pretrained(
            str(model_path), local_files_only=True
        )
        if vision_tower:
            # Fix the legacy loader that retained a stale absolute/HF tower in
            # config.json even when a local --vision-tower was supplied.
            for name in ("mm_vision_tower", "vision_tower"):
                if hasattr(config, name) or name == "mm_vision_tower":
                    setattr(config, name, str(vision_tower))

        model = StreamVLNForCausalLM.from_pretrained(
            str(model_path),
            attn_implementation=attention_implementation,
            torch_dtype=dtype_value,
            config=config,
            low_cpu_mem_usage=False,
            local_files_only=True,
        )
        if hasattr(model, "model"):
            model.model.num_history = num_history
        model.requires_grad_(False)

        loaded_vision_tower = model.get_vision_tower()
        if not getattr(loaded_vision_tower, "is_loaded", False):
            loaded_vision_tower.load_model()
        loaded_vision_tower.to(device=torch_device, dtype=dtype_value)
        model.to(torch_device)
        model.eval()
        return cls(
            model=model,
            tokenizer=tokenizer,
            image_processor=loaded_vision_tower.image_processor,
            torch_module=torch,
            dict_to_device=dict_to_cuda,
            image_token_index=IMAGE_TOKEN_INDEX,
            memory_token_index=MEMORY_TOKEN_INDEX,
            device=torch_device,
            image_dtype=dtype_value,
            num_frames=num_frames,
            num_history=num_history,
            num_future_steps=num_future_steps,
            max_new_tokens=max_new_tokens,
        )

    def reset(self, context: EpisodeContext) -> None:
        """Reset per-episode chunk, KV cache, model env, and history state."""

        if self._closed:
            raise RuntimeError("StreamVLNPolicyAdapter is closed")
        self._context = context
        self._instruction = episode_instruction(context.episode)
        self._action_queue.clear()
        self._output_ids = None
        self._past_key_values = None
        self.history.reset()
        self._rng = random.Random(context.seed)
        if self._model_world_size != context.world_size:
            self.model.reset(context.world_size)
            self._model_world_size = context.world_size
        self.model.reset_for_env(context.rank)
        self.model.eval()

    def _process_rgb(self, rgb: Any) -> Any:
        from PIL import Image

        if isinstance(rgb, Image.Image):
            image = rgb.convert("RGB")
        else:
            image = Image.fromarray(rgb).convert("RGB")
        return self.image_processor.preprocess(
            images=image, return_tensors="pt"
        )["pixel_values"][0]

    def _preprocess_qwen(self, sources: List[List[Mapping[str, str]]], add_system: bool):
        roles = {"human": "user", "gpt": "assistant"}
        tokenizer = copy.deepcopy(self.tokenizer)
        tokenizer.add_tokens([DEFAULT_IMAGE_TOKEN], special_tokens=True)
        tokenizer.add_tokens([DEFAULT_MEMORY_TOKEN], special_tokens=True)
        image_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
        memory_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_MEMORY_TOKEN)
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] "
            "+ '<|im_end|>' + '\\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
        )

        input_ids = []
        for source in sources:
            source = copy.deepcopy(source)
            prompt = self._rng.choice(self.CONJUNCTIONS) + DEFAULT_IMAGE_TOKEN
            source[0]["value"] = (
                source[0]["value"] + f" {prompt}."
                if source[0]["value"]
                else f"{prompt}."
            )
            if roles[source[0]["from"]] != roles["human"]:
                source = source[1:]
            encoded = []
            if add_system:
                encoded += tokenizer.apply_chat_template(
                    [{"role": "system", "content": "You are a helpful assistant."}]
                )
            for message in source:
                role = roles.get(message.get("from", ""), message.get("role", ""))
                content = message.get("value", message.get("content", ""))
                encoded += tokenizer.apply_chat_template(
                    [{"role": role, "content": content}]
                )
            encoded = [
                self.image_token_index
                if token == image_token_id
                else self.memory_token_index
                if token == memory_token_id
                else token
                for token in encoded
            ]
            input_ids.append(encoded)
        return self.torch.tensor(input_ids, dtype=self.torch.long)

    def _generate_action_chunk(self) -> List[int]:
        if self._context is None:
            raise RuntimeError("reset() must be called before act()")
        first_in_window = self._output_ids is None
        if first_in_window:
            sources = [
                {"from": "human", "value": self.PROMPT},
                {"from": "gpt", "value": ""},
            ]
            sources[0]["value"] = sources[0]["value"].replace(
                DEFAULT_VIDEO_TOKEN + "\n", ""
            )
            if self.history.step_id != 0:
                sources[0]["value"] += (
                    f" These are your historical observations {DEFAULT_MEMORY_TOKEN}."
                )
            sources[0]["value"] = sources[0]["value"].replace(
                "<instruction>.", self._instruction
            )
        else:
            sources = [
                {"from": "human", "value": ""},
                {"from": "gpt", "value": ""},
            ]

        input_ids = self._preprocess_qwen(
            [sources], add_system=first_in_window
        )
        if self._output_ids is not None:
            input_ids = self.torch.cat(
                [self._output_ids, input_ids.to(self._output_ids.device)], dim=1
            )

        images = self.history.generation_frames(
            first_generation_in_window=first_in_window,
            num_history=self.num_history,
            future_stride=self.num_future_steps,
        )
        image_tensor = self.torch.stack(images).unsqueeze(0)
        model_inputs = {
            "images": image_tensor,
            "depths": None,
            "poses": None,
            "intrinsics": None,
            "inputs": input_ids,
            "env_id": self._context.rank,
            "time_ids": [list(self.history.window_time_ids)],
            "task_type": [0],
        }
        model_inputs = self.dict_to_device(model_inputs, self.device)
        if model_inputs["images"] is not None:
            model_inputs["images"] = model_inputs["images"].to(self.image_dtype)

        with self.torch.inference_mode():
            outputs = self.model.generate(
                **model_inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                return_dict_in_generate=True,
                past_key_values=self._past_key_values,
            )
        self._output_ids = outputs.sequences
        self._past_key_values = outputs.past_key_values
        decoded = self.tokenizer.batch_decode(
            self._output_ids, skip_special_tokens=False
        )[0].strip()
        actions = parse_action_symbols(decoded)
        return actions or [0]

    def act(self, observation: Mapping[str, Any]) -> PolicyStep:
        """Return one primitive action, retaining any remaining action chunk."""

        if self._context is None:
            raise RuntimeError("reset() must be called before act()")
        if "rgb" not in observation:
            raise KeyError("StreamVLN requires observation['rgb']")
        self.history.observe(self._process_rgb(observation["rgb"]))
        generated = False
        if not self._action_queue:
            self._action_queue.extend(self._generate_action_chunk())
            generated = True
        action = self._action_queue.pop(0)
        step_id = self.history.step_id
        window_reset = self.history.action_executed()
        if window_reset:
            # Keep queued actions and global frames.  Only the 32-frame model
            # window and its KV/text state reset.
            self.model.reset_for_env(self._context.rank)
            self._output_ids = None
            self._past_key_values = None
        return PolicyStep(
            action=action,
            info={
                "step_id": step_id,
                "generated_chunk": generated,
                "chunk_remaining": len(self._action_queue),
                "window_reset": window_reset,
            },
        )

    def close(self) -> None:
        self._action_queue.clear()
        self._output_ids = None
        self._past_key_values = None
        self._context = None
        self._closed = True
