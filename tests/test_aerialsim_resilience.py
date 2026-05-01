import numpy as np

from satnav.sims.aerialsim.aerialsim import AerialSim, RenderValidationError


def make_sim(aerial_overrides=None) -> AerialSim:
    aerial_config = {
        "API_KEY": "dummy",
        "ENABLE_API_PREFLIGHT": False,
    }
    if aerial_overrides:
        aerial_config.update(aerial_overrides)
    return AerialSim(
        {
            "FORWARD_STEP_SIZE": 10.0,
            "TURN_ANGLE": 15.0,
            "RGB_SENSOR": {"WIDTH": 64, "HEIGHT": 64, "HFOV": 90.0},
            "AERIAL": aerial_config,
        }
    )


def make_colorful_metrics(sim: AerialSim):
    xx, yy = np.meshgrid(np.arange(64, dtype=np.uint8), np.arange(64, dtype=np.uint8))
    rgb = np.stack((xx, yy, ((xx.astype(np.uint16) + yy.astype(np.uint16)) % 255).astype(np.uint8)), axis=2)
    return sim.compute_image_quality_metrics(rgb)


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
    metrics = make_colorful_metrics(sim)
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


def test_salvage_quality_gate_allows_configured_pending_work() -> None:
    sim = make_sim(
        {
            "ALLOW_SALVAGE_CAPTURE": True,
            "SALVAGE_MAX_PENDING_REQUESTS": 3,
            "SALVAGE_MAX_TILES_PROCESSING": 2,
        }
    )
    metrics = make_colorful_metrics(sim)
    status = {
        "ready": True,
        "render_ready": True,
        "tiles_loaded": 32,
        "pending_requests": 3,
        "tiles_processing": 2,
    }

    assert sim._can_salvage_capture_from_status(status)
    assert sim._get_quality_failure_reasons(metrics, status) != []
    assert sim._get_quality_failure_reasons(
        metrics,
        status,
        allow_salvage_status=True,
    ) == []

    over_limit_status = {**status, "pending_requests": 4}
    over_limit_reasons = sim._get_quality_failure_reasons(
        metrics,
        over_limit_status,
        allow_salvage_status=True,
    )
    assert any("pending_requests=4 > 3" in reason for reason in over_limit_reasons)


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


def test_public_diagnostics_and_ground_cache_threshold() -> None:
    sim = make_sim()
    sim.set_ground_height_cache_threshold(12.5)

    diagnostics = sim.get_diagnostics()

    assert diagnostics["local_height_range"] == 0.0
    assert diagnostics["ground_height_cache_threshold"] == 12.5


def test_chrome_remote_debugging_port_defaults_to_auto_port() -> None:
    sim = make_sim()
    assert sim._get_chrome_remote_debugging_argument() == "--remote-debugging-port=0"

    configured = make_sim({"CHROME_REMOTE_DEBUGGING_PORT": 9333})
    assert configured._get_chrome_remote_debugging_argument() == "--remote-debugging-port=9333"
