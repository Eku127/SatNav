from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from .configuration_prismatic import OpenFlyConfig
from .modeling_prismatic import OpenVLAForActionPrediction
from .processing_prismatic import PrismaticImageProcessor, PrismaticProcessor


def register_openfly_auto_classes() -> None:
    """Register OpenFly HF components for local checkpoint loading."""
    try:
        AutoConfig.register("openvla", OpenFlyConfig)
    except ValueError:
        pass

    try:
        AutoImageProcessor.register(OpenFlyConfig, PrismaticImageProcessor)
    except ValueError:
        pass

    try:
        AutoProcessor.register(OpenFlyConfig, PrismaticProcessor)
    except ValueError:
        pass

    try:
        AutoModelForVision2Seq.register(OpenFlyConfig, OpenVLAForActionPrediction)
    except ValueError:
        pass
