from hydra.core.config_search_path import ConfigSearchPath
from hydra.core.config_store import ConfigStore
from hydra.plugins.search_path_plugin import SearchPathPlugin
from habitat.config.default_structured_configs import LabSensorConfig
from habitat_baselines.config.default_structured_configs import HabitatBaselinesBaseConfig, ActionDistributionConfig, HierarchicalPolicyConfig, ObsTransformConfig
from typing import Dict

from omegaconf import MISSING
from dataclasses import dataclass, field

cs = ConfigStore.instance()

# TODO: Make this customizable
# @dataclass
# class CustomPolicyConfig(HabitatBaselinesBaseConfig):
#     name: str = "PointNavVC1PolicyConfig"
#     action_distribution_type: str = "categorical"  # or 'gaussian'
#     # If the list is empty, all keys will be included.
#     # For gaussian action distribution:
#     action_dist: ActionDistributionConfig = ActionDistributionConfig()
#     obs_transforms: Dict[str, ObsTransformConfig] = field(default_factory=dict)
#     hierarchical_policy: HierarchicalPolicyConfig = MISSING
#     fp_16: bool = False


@dataclass
class CacheImageGoalSensorConfig(LabSensorConfig):
    type: str = "CacheImageGoalSensor"
    cache: str = "/srv/flash1/gchhablani3/spring_2024/vlm-task/data/datasets/vc1_embeddings/"

cs.store(
    package=f"habitat.task.lab_sensors.cache_imagegoal_sensor",
    group="habitat/task/lab_sensors",
    name="cache_imagegoal_sensor",
    node=CacheImageGoalSensorConfig,
)

class HabitatConfigPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(
            provider="habitat",
            path="pkg://config/experiments/",
        )