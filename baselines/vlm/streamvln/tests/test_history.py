from baselines.vlm.streamvln.history import StreamingHistory, history_indices


def test_history_indices_match_legacy_sampling():
    assert history_indices(32, num_history=8, future_stride=4) == tuple(range(0, 32, 4))
    assert history_indices(9, num_history=8, future_stride=4) == tuple(range(9))
    assert history_indices(10, num_history=None, future_stride=4) == (0, 4, 8)


def test_window_boundary_resets_time_ids_but_keeps_global_frames():
    history = StreamingHistory(num_frames=4)
    for step in range(4):
        history.observe(f"frame-{step}")
        assert history.action_executed() is (step == 3)

    assert history.step_id == 4
    assert history.frames == ["frame-0", "frame-1", "frame-2", "frame-3"]
    assert history.window_time_ids == []


def test_leftover_chunk_crossing_window_anchors_history_at_boundary():
    history = StreamingHistory(num_frames=4)
    for step in range(4):
        history.observe(f"frame-{step}")
        history.action_executed()

    # A queued action consumes step 4 without generating.  Generation at step
    # 5 must use window_time_ids[0] == 4, plus the latest frame-5.
    history.observe("frame-4")
    history.action_executed()
    history.observe("frame-5")
    assert history.generation_frames(
        first_generation_in_window=True,
        num_history=2,
        future_stride=2,
    ) == ["frame-0", "frame-2", "frame-5"]
