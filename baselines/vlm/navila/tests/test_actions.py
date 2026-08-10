from PIL import Image

from baselines.vlm.navila.actions import (
    build_prompt,
    normalize_instruction,
    parse_action_queue,
    sample_and_pad_images,
)


def test_instruction_prompt_and_compact_action_queue():
    assert normalize_instruction("go north .\nthen STOP") == "Go north. Then stop"
    prompt = build_prompt("go north", 7, "compact")
    assert prompt.count("<image>") == 8
    assert parse_action_queue("forward") == (1, [])
    assert parse_action_queue("turn left 45 degrees") == (2, [2, 2])
    assert parse_action_queue("move forward 30 meters") == (1, [1, 1])
    assert parse_action_queue("unparseable") == (1, [])


def test_image_sampling_retains_latest_and_pads_to_eight():
    image = Image.new("RGB", (4, 4), color=(1, 2, 3))
    sampled = sample_and_pad_images([image], num_frames=8, width=4, height=4)
    assert len(sampled) == 8
    assert sampled[-1].getpixel((0, 0)) == (1, 2, 3)
    assert sampled[0].getpixel((0, 0)) == (0, 0, 0)
