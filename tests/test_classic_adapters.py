import sys
import unittest
from dataclasses import dataclass


@dataclass
class _State:
    position: list
    rotation: float = 0.0


class _Simulator:
    def __init__(self):
        self.state = _State([0.0, 0.0, 0.0])

    def get_agent_state(self):
        return self.state


class _Environment:
    def __init__(self):
        self.simulator = _Simulator()

    @property
    def agent_state(self):
        return self.simulator.get_agent_state()


class _Episode:
    episode_id = "7"
    scene_id = "scene"
    start_position = [0.0, 0.0, 0.0]
    reference_path = [[0.0, 0.0, 0.0], [0.001, 0.0, 0.0]]
    goals = []


class ClassicAdapterTests(unittest.TestCase):
    @staticmethod
    def _context(seed=123):
        from satnav.evaluation import EpisodeContext

        return EpisodeContext(
            episode=_Episode(),
            episode_key="val_seen::scene::7",
            split="val_seen",
            episode_index=0,
            rank=0,
            world_size=1,
            max_steps=10,
            seed=seed,
            environment=_Environment(),
        )

    def test_import_is_torch_free(self):
        for name in list(sys.modules):
            if name == "baselines" or name.startswith("baselines.classic"):
                del sys.modules[name]
        before = "torch" in sys.modules
        import baselines.classic  # noqa: F401
        self.assertEqual("torch" in sys.modules, before)

    def test_random_is_episode_seeded(self):
        from baselines.classic.agents import RandomAdapter

        adapter = RandomAdapter()
        context = self._context(seed=991)
        adapter.reset(context)
        first = [adapter.act({}).action for _ in range(12)]
        adapter.reset(context)
        second = [adapter.act({}).action for _ in range(12)]
        self.assertEqual(first, second)

    def test_reference_follower_uses_public_simulator(self):
        from baselines.classic.agents import ReferenceFollowerAdapter
        from satnav.task.actions import Action

        adapter = ReferenceFollowerAdapter(goal_radius=3.0)
        adapter.reset(self._context())
        step = adapter.act({})
        self.assertIn(step.action, Action.ALL_ACTIONS)
        self.assertEqual(step.info["policy"], "reference_follower")


if __name__ == "__main__":
    unittest.main()
