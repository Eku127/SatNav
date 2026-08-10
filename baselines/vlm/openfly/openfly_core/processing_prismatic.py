"""
processing_prismatic.py

HuggingFace-style preprocessor definitions for Prismatic VLMs, inheriting from `ProcessorMixin`. Default configuration
specifies `siglip-224px+7b`.
"""

import math
from typing import Any, ClassVar, List, Optional, Tuple, Union

import timm.data
import torch
import torchvision.transforms.functional as TVF
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from transformers import PreTrainedTokenizerBase
from transformers.image_processing_utils import BatchFeature, ImageProcessingMixin
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils import PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from transformers.utils import TensorType


# === Image Processing ===
def letterbox_pad_transform(image: Image.Image, padding_fill_value: Tuple[int, int, int]) -> Image.Image:
    """Given a PIL.Image, pad to square by adding a symmetric border around the height/width."""
    (w, h), max_wh = image.size, max(image.size)
    horizontal_total, vertical_total = max_wh - w, max_wh - h
    left, top = horizontal_total // 2, vertical_total // 2
    padding = (left, top, horizontal_total - left, vertical_total - top)

    return TVF.pad(image, padding, fill=padding_fill_value, padding_mode="constant")


class PrismaticImageProcessor(ImageProcessingMixin):
    model_input_names: ClassVar[List[str]] = ["pixel_values"]

    def __init__(
        self,
        use_fused_vision_backbone: bool = False,
        image_resize_strategy: str = "letterbox",
        input_sizes: Optional[List[Tuple[int, int, int]]] = None,
        interpolations: Optional[List[str]] = None,
        means: Optional[List[Tuple[float, float, float]]] = None,
        stds: Optional[List[Tuple[float, float, float]]] = None,
        **kwargs: str,
    ) -> None:
        """
        Initialize a PrismaticImageProcessor as a wrapper around a torchvision transform; this transform will be
        created by TIMM, and edited to follow our custom `image_resize_strategy` logic.
        @param use_fused_vision_backbone: Boolean indicating single or fused (dual) vision backbone
        @param image_resize_strategy: Prismatic image resize strategy in < resize-naive | resize-crop | letterbox >
        @param input_size: [TIMM :: `data_cfg`] Input image size as tuple (channels, width, height)
        @param interpolation: [TIMM :: `data_cfg`] Interpolation as string (default: "bicubic")
        @param mean: [TIMM :: `data_cfg`] Normalization mean as float tuple (or two-tuple if `fused_backbone`)
        @param std: [TIMM :: `data_cfg`] Normalization std as float tuple (or two-tuple if `fused_backbone`)
        """
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.image_resize_strategy = image_resize_strategy

        # Handle `None` default values
        input_sizes = [(3, 224, 224)] if input_sizes is None else input_sizes
        interpolations = (
            ["bicubic"] * len(input_sizes)
            if interpolations is None
            else interpolations
        )
        means = [(0.5, 0.5, 0.5)] if means is None else means
        stds = [(0.5, 0.5, 0.5)] if stds is None else stds
        if not (
            len(input_sizes)
            == len(interpolations)
            == len(means)
            == len(stds)
        ):
            raise ValueError("OpenFly image processor configuration lengths differ")

        # TIMM `data_cfg` Parameters
        self.input_sizes, self.interpolations, self.means, self.stds = input_sizes, interpolations, means, stds

        # Grab torchvision transforms via TIMM =>> need to parse for specific "functional" transform values!
        self.tvf_resize_params, self.tvf_crop_params, self.tvf_normalize_params = [], [], []
        self.tvf_do_letterbox, self.tvf_letterbox_fill = [], []

        for idx in range(len(input_sizes)):
            transform = timm.data.create_transform(
                input_size=self.input_sizes[idx],
                interpolation=self.interpolations[idx],
                mean=self.means[idx],
                std=self.stds[idx],
                crop_pct=1.0,  # Set to 1.0 to ignore cropping (initial Resize sets `input_size`)
                crop_mode="center",  # Default crop mode -- no-op when `crop_pct == 1.0`
                is_training=False,  # No image augmentations when loading the transform!
            )

            # [Validation] Ensure appropriate transform structure, expected sizes
            if not (
                isinstance(transform, Compose)
                and (len(transform.transforms) == 4)
                and isinstance(transform.transforms[0], Resize)
                and isinstance(transform.transforms[1], CenterCrop)
                and isinstance(transform.transforms[2], ToTensor)
                and isinstance(transform.transforms[3], Normalize)
                and (transform.transforms[0].size == self.input_sizes[idx][-1])
                and (transform.transforms[1].size == self.input_sizes[idx][-2:])
            ):
                raise ValueError(f"Unexpected TIMM image transformation structure/sizes: `{transform}`")

            # HF Image Processors *must* be JSON-serializable; as such, cannot have torchvision. as an attribute.
            #   => Instead, we're going to parse the transform and call "torchvision.transforms.functional" (`tvf`)
            resize_t, crop_t, norm_t = transform.transforms[0], transform.transforms[1], transform.transforms[3]
            self.tvf_resize_params.append(
                {
                    "size": resize_t.size,
                    "interpolation": TVF.pil_modes_mapping[resize_t.interpolation],
                    "max_size": None,
                    "antialias": True,
                }
            )
            self.tvf_crop_params.append({"output_size": crop_t.size})
            self.tvf_normalize_params.append(
                {
                    "mean": norm_t.mean.float().numpy().tolist(),
                    "std": norm_t.std.float().numpy().tolist(),
                    "inplace": False,
                }
            )
            do_letterbox, letterbox_fill = False, None

            # Handle Prismatic `image_resize_strategy`
            if self.image_resize_strategy == "resize-naive":
                self.tvf_resize_params[idx]["size"] = (resize_t.size, resize_t.size)
            elif self.image_resize_strategy == "letterbox":
                do_letterbox = True
                letterbox_fill = tuple(int(x * 255) for x in self.means[idx])
            elif self.image_resize_strategy == "resize-crop":
                pass
            else:
                raise ValueError(f"Image resize strategy `{self.image_resize_strategy}` is not supported!")
            self.tvf_do_letterbox.append(do_letterbox)
            self.tvf_letterbox_fill.append(letterbox_fill)

        # Dispatch **kwargs to super()
        super().__init__(**kwargs)
        self._normalize_and_validate_transform_state()

    def _normalize_and_validate_transform_state(self) -> None:
        """Validate serialized transform caches and migrate legacy resize state.

        Older OpenFly processors serialized letterbox state as one scalar even
        for fused backbones.  The other serialized transform caches are kept
        verbatim because their numeric values are part of checkpoint behavior.
        """
        backbone_count = len(self.input_sizes)
        if backbone_count == 0:
            raise ValueError("OpenFly image processor requires a vision backbone")

        for attribute in (
            "tvf_resize_params",
            "tvf_crop_params",
            "tvf_normalize_params",
        ):
            cache = getattr(self, attribute, None)
            if not (
                isinstance(cache, list)
                and len(cache) == backbone_count
                and all(isinstance(item, dict) for item in cache)
            ):
                raise ValueError(
                    f"OpenFly image processor has invalid {attribute}: expected "
                    f"{backbone_count} dictionaries"
                )

        allowed_interpolations = set(TVF.pil_modes_mapping.values())
        for index, input_size in enumerate(self.input_sizes):
            if not (
                isinstance(input_size, (list, tuple))
                and len(input_size) == 3
                and all(type(value) is int and value > 0 for value in input_size)
            ):
                raise ValueError("OpenFly input_sizes must contain positive CHW sizes")

            resize = self.tvf_resize_params[index]
            if set(resize) != {"size", "interpolation", "max_size", "antialias"}:
                raise ValueError(
                    "OpenFly tvf_resize_params entries have invalid keys"
                )
            resize_size = self._positive_size_tuple(resize["size"])
            expected_resize = (
                (input_size[-1], input_size[-1])
                if self.image_resize_strategy == "resize-naive"
                else (input_size[-1],)
            )
            if resize_size != expected_resize:
                raise ValueError(
                    "OpenFly tvf_resize_params size contradicts input_sizes"
                )
            if not (
                type(resize["interpolation"]) is int
                and resize["interpolation"] in allowed_interpolations
                and resize["max_size"] is None
                and type(resize["antialias"]) is bool
            ):
                raise ValueError("OpenFly tvf_resize_params values are invalid")

            crop = self.tvf_crop_params[index]
            if set(crop) != {"output_size"}:
                raise ValueError("OpenFly tvf_crop_params entries have invalid keys")
            if self._positive_size_tuple(crop["output_size"]) != tuple(
                input_size[-2:]
            ):
                raise ValueError(
                    "OpenFly tvf_crop_params output_size contradicts input_sizes"
                )

            normalize = self.tvf_normalize_params[index]
            if set(normalize) != {"mean", "std", "inplace"}:
                raise ValueError(
                    "OpenFly tvf_normalize_params entries have invalid keys"
                )
            channels = input_size[0]
            mean, std = normalize["mean"], normalize["std"]
            if not (
                isinstance(mean, (list, tuple))
                and isinstance(std, (list, tuple))
                and len(mean) == len(std) == channels
                and all(self._is_finite_float32_number(value) for value in mean)
                and all(
                    self._is_positive_finite_float32_number(value) for value in std
                )
                and type(normalize["inplace"]) is bool
            ):
                raise ValueError("OpenFly tvf_normalize_params values are invalid")

        flags = self.tvf_do_letterbox
        fills = self.tvf_letterbox_fill
        if type(flags) is bool:
            if not (
                flags is False
                and fills is None
                and self.image_resize_strategy in {"resize-naive", "resize-crop"}
            ):
                raise ValueError(
                    "Unsupported legacy OpenFly letterbox metadata; only "
                    "false/null non-letterbox state can be migrated losslessly"
                )
            flags = [False] * backbone_count
            fills = [None] * backbone_count

        if not (
            isinstance(flags, list)
            and len(flags) == backbone_count
            and all(type(flag) is bool for flag in flags)
        ):
            raise ValueError(
                "OpenFly tvf_do_letterbox must be a bool list matching the "
                "number of vision backbones"
            )
        if not isinstance(fills, list) or len(fills) != backbone_count:
            raise ValueError(
                "OpenFly tvf_letterbox_fill must be a list matching the "
                "number of vision backbones"
            )

        expected_flag = self.image_resize_strategy == "letterbox"
        if self.image_resize_strategy not in {
            "letterbox",
            "resize-naive",
            "resize-crop",
        } or any(flag is not expected_flag for flag in flags):
            raise ValueError(
                "OpenFly letterbox metadata contradicts image_resize_strategy"
            )
        for flag, fill in zip(flags, fills):
            if not flag:
                if fill is not None:
                    raise ValueError(
                        "OpenFly non-letterbox transform must have a null fill"
                    )
                continue
            if not (
                isinstance(fill, (list, tuple))
                and len(fill) == 3
                and all(
                    type(component) is int and 0 <= component <= 255
                    for component in fill
                )
            ):
                raise ValueError(
                    "OpenFly letterbox fill must contain three integer RGB values"
                )

        self.tvf_do_letterbox = flags
        self.tvf_letterbox_fill = fills

    @staticmethod
    def _positive_size_tuple(value: Any) -> Tuple[int, ...]:
        values = (value,) if type(value) is int else value
        if not (
            isinstance(values, (list, tuple))
            and len(values) in {1, 2}
            and all(type(item) is int and item > 0 for item in values)
        ):
            raise ValueError("OpenFly transform size must contain positive integers")
        return tuple(values)

    @staticmethod
    def _is_finite_float32_number(value: Any) -> bool:
        if type(value) not in {int, float} or not math.isfinite(value):
            return False
        converted = torch.tensor(value, dtype=torch.float32).item()
        return math.isfinite(converted)

    @classmethod
    def _is_positive_finite_float32_number(cls, value: Any) -> bool:
        if not cls._is_finite_float32_number(value):
            return False
        return torch.tensor(value, dtype=torch.float32).item() > 0

    def apply_transform(self, img: Image.Image) -> torch.Tensor:
        """Apply `functional` variant of TIMM's Transform = Compose([Resize -> CenterCrop -> ToTensor -> Normalize])"""
        # [Contract] Fused Backbones expect "channel-stacked" inputs; we'll unpack on the model side!
        imgs_t = []
        for idx in range(len(self.input_sizes)):
            img_idx = img
            if self.tvf_do_letterbox[idx]:
                img_idx = letterbox_pad_transform(
                    img_idx, self.tvf_letterbox_fill[idx]
                )
            img_idx = TVF.resize(img_idx, **self.tvf_resize_params[idx])
            img_idx = TVF.center_crop(img_idx, **self.tvf_crop_params[idx])
            img_idx_t = TVF.to_tensor(img_idx)
            img_idx_t = TVF.normalize(img_idx_t, **self.tvf_normalize_params[idx])
            imgs_t.append(img_idx_t)

        # [Contract] `imgs_t` is a list of Tensors of shape [3, input_size, input_size]; stack along dim = 0
        img_t = torch.vstack(imgs_t)

        return img_t

    def preprocess(
        self,
        images: Union[Image.Image, List[Image.Image]],
        return_tensors: Optional[Union[str, TensorType]] = None,
        **_: str,
    ) -> BatchFeature:
        """
        Preprocess an image (or batch of images); note that unlike the `transformers :: BaseImageProcessor` we
        explicitly only handle PIL.Image.Image instances for simplicity.
        @param images: A (batch of) PIL.Image.Image instance(s) to preprocess.
        @param return_tensors: BatchFeature default Tensor format (e.g., "pt" for torch); if None, returns np.ndarray
        @return: Instance of `transformers :: BatchFeature` with a single key "pixel_values"
        """
        if not isinstance(images, list):
            images = [images]

        # Apply `self.img_transform` to each image (will return list of torch.Tensors); stack into "batched" Tensor
        pixel_values = torch.stack([self.apply_transform(img.convert("RGB")) for img in images])

        # Return BatchFeature =>> note that for compatibility, constructor expects Dict[str, np.ndarray], so we convert
        return BatchFeature(data={"pixel_values": pixel_values.float().numpy()}, tensor_type=return_tensors)

    def __call__(self, images: Union[Image.Image, List[Image.Image]], **kwargs) -> BatchFeature:
        return self.preprocess(images, **kwargs)


# === PrismaticProcessor =>> Wraps both ImageProcessor and Tokenizer ===
#   =>> https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava/processing_llava.py
class PrismaticProcessor(ProcessorMixin):
    attributes: ClassVar[List[str]] = ["image_processor", "tokenizer"]
    image_processor_class: str = "AutoImageProcessor"
    tokenizer_class: str = "AutoTokenizer"

    def __init__(
        self,
        image_processor: Optional[ImageProcessingMixin] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ) -> None:
        super().__init__(image_processor, tokenizer)

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]],
        images: Union[Image.Image, List[Image.Image]],
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Optional[Union[bool, str, TruncationStrategy]] = None,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
    ) -> BatchFeature:
        """
        Preprocess a given (batch) of text/images for a Prismatic VLM; forwards text to the underlying LLM's tokenizer,
        forwards images to PrismaticImageProcessor.
        @param text: The (batch) of text to encode; must be a string or list of strings.
        @param images: A (batch of) PIL.Image.Image instance(s) to preprocess.
        @param padding: Sequence padding strategy (if multiple specified) in < True = "longest" | "max_length" | False >
        @param truncation: Truncation strategy for the output sequences; requires `max_length` to be specified
        @param max_length: Maximum length (in tokens) to truncate
        @param return_tensors: Type of return tensors (usually "pt" or TensorType.PYTORCH)
        @return: BatchFeature with keys for `input_ids`, `attention_mask` and `pixel_values`.
        """
        pixel_values = self.image_processor(images, return_tensors=return_tensors)["pixel_values"]
        text_inputs = self.tokenizer(
            text, return_tensors=return_tensors, padding=padding, truncation=truncation, max_length=max_length
        )

        return BatchFeature(data={**text_inputs, "pixel_values": pixel_values})

    # === Tokenizer Dispatch Utilities =>> check `PreTrainedTokenizerBase` for documentation ===
    def batch_decode(
        self,
        sequences: Union[List[int], List[List[int]], torch.Tensor, Any],  # `Any` = np.ndarray | tf.Tensor
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: Optional[bool] = None,
        **kwargs: str,
    ) -> List[str]:
        return self.tokenizer.batch_decode(
            sequences=sequences,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    def decode(
        self,
        token_ids: Union[int, List[int], torch.Tensor, Any],  # `Any` = np.ndarray | tf.Tensor
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: Optional[bool] = None,
        **kwargs: str,
    ) -> str:
        return self.tokenizer.decode(
            token_ids=token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    @property
    def model_input_names(self) -> List[str]:
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names

        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
