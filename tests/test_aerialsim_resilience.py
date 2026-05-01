import numpy as np

from satnav.sims.aerialsim.aerialsim import AerialSim, RenderValidationError


def make_sim() -> AerialSim:
    return AerialSim(
        {
            "FORWARD_STEP_SIZE": 10.0,
            "TURN_ANGLE": 15.0,
            "RGB_SENSOR": {"WIDTH": 64, "HEIGHT": 64, "HFOV": 90.0},
            "AERIAL": {
                "API_KEY": "dummy",
                "ENABLE_API_PREFLIGHT": False,
            },
        }
    )


def test_quality_gate_rejects_nearly_black_render() -> None:
    sim = make_sim()
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    metrics = sim.compute_image_quality_metrics(rgb)
    reasons = sim._get_quality_failure_reasons(
        metrics,
        {
            "ready": True,
            "render_ready": True,
            "tiles_loaded": 16,
            "pending_requests": 0,
            "tiles_processing": 0,
        },
    )

    assert any("black_frac" in reason for reason in reasons)
    assert any("mean=" in reason for reason in reasons)
    assert any("std=" in reason for reason in reasons)


def test_quality_gate_accepts_colorful_render() -> None:
    sim = make_sim()
    xx, yy = np.meshgrid(np.arange(64, dtype=np.uint8), np.arange(64, dtype=np.uint8))
    rgb = np.stack((xx, yy, ((xx.astype(np.uint16) + yy.astype(np.uint16)) % 255).astype(np.uint8)), axis=2)
    metrics = sim.compute_image_quality_metrics(rgb)
    reasons = sim._get_quality_failure_reasons(
        metrics,
        {
            "ready": True,
            "render_ready": True,
            "tiles_loaded": 32,
            "pending_requests": 0,
            "tiles_processing": 0,
        },
    )

    assert reasons == []


def test_status_gate_requires_ready_tiles_and_no_pending_requests() -> None:
    sim = make_sim()

    assert sim._is_status_ready_for_capture(
        {
            "ready": True,
            "render_ready": True,
            "stable": True,
            "tiles_loaded": 8,
            "pending_requests": 0,
            "tiles_processing": 0,
        },
        min_tiles_loaded=8,
    )

    assert not sim._is_status_ready_for_capture(
        {
            "ready": True,
            "render_ready": True,
            "stable": True,
            "tiles_loaded": 4,
            "pending_requests": 0,
            "tiles_processing": 0,
        },
        min_tiles_loaded=8,
    )

    assert not sim._is_status_ready_for_capture(
        {
            "ready": True,
            "render_ready": True,
            "stable": True,
            "tiles_loaded": 32,
            "pending_requests": 1,
            "tiles_processing": 0,
        },
        min_tiles_loaded=8,
    )


def test_restart_policy_triggers_after_repeated_validation_failures() -> None:
    sim = make_sim()
    sim._consecutive_capture_failures = sim._restart_after_failures
    assert sim._should_restart_browser(RenderValidationError("bad frame"))
