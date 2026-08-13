"""Tests for built-in and external simulator construction."""

import unittest
from unittest import mock

from omegaconf import OmegaConf

from satnav.core.simulator import AgentState, Simulator
from satnav.sims import create_simulator


class ExternalSimulator(Simulator):
    def __init__(self, config, scenes_dir):
        self.config = config
        self.scenes_dir = scenes_dir

    def reset(self, scene_id):
        return {}

    def step(self, action):
        return {}

    def get_agent_state(self):
        return AgentState([0.0, 0.0, 0.0], 0.0)

    def set_agent_state(self, position, rotation):
        return None

    def get_observations(self):
        return {}

    def geodesic_distance(self, position_a, position_b):
        return 0.0

    def is_navigable(self, position):
        return True

    @property
    def sensor_suite(self):
        return {}

    @property
    def action_space(self):
        return []


class NotASimulator:
    pass


class SimulatorFactoryTests(unittest.TestCase):
    def test_external_class_receives_full_config_and_scene_root(self):
        class_path = f"{__name__}.ExternalSimulator"
        config = {
            "SIMULATOR": {"TYPE": "external", "CLASS": class_path}
        }

        simulator = create_simulator(config, scenes_dir="/scenes")

        self.assertIsInstance(simulator, ExternalSimulator)
        self.assertIs(simulator.config, config)
        self.assertEqual(simulator.scenes_dir, "/scenes")

    def test_external_class_supports_dictconfig(self):
        class_path = f"{__name__}.ExternalSimulator"
        config = OmegaConf.create(
            {"SIMULATOR": {"TYPE": "external", "CLASS": class_path}}
        )
        simulator = create_simulator(config, scenes_dir="/scenes")
        self.assertIsInstance(simulator, ExternalSimulator)

    def test_unknown_type_requires_a_class(self):
        with self.assertRaisesRegex(ValueError, "SIMULATOR.CLASS"):
            create_simulator({"SIMULATOR": {"TYPE": "external"}})

    def test_external_class_must_inherit_simulator(self):
        class_path = f"{__name__}.NotASimulator"
        with self.assertRaisesRegex(TypeError, "must inherit"):
            create_simulator(
                {"SIMULATOR": {"TYPE": "external", "CLASS": class_path}}
            )

    def test_missing_external_class_has_clear_error(self):
        with self.assertRaisesRegex(ImportError, "was not found"):
            create_simulator(
                {
                    "SIMULATOR": {
                        "TYPE": "external",
                        "CLASS": f"{__name__}.MissingSimulator",
                    }
                }
            )

    def test_satsim_default_remains_the_builtin_path(self):
        sentinel = object()
        constructor = mock.Mock(return_value=sentinel)
        with mock.patch("satnav.sims._get_satsim_wrapper", return_value=constructor):
            result = create_simulator({"SIMULATOR": {}}, scenes_dir="/scenes")
        self.assertIs(result, sentinel)
        constructor.assert_called_once()


if __name__ == "__main__":
    unittest.main()
