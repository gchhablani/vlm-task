import argparse
import os

import habitat
import numpy as np
import torch
from habitat.config import read_write
from habitat_baselines.config.default import get_config

import pickle

import numpy as np
import torch
from vc_models.models.vit import model_utils
from tqdm import tqdm

class VC1Encoder:
    def __init__(
        self, name: str = model_utils.VC1_LARGE_NAME, device: str = "cuda"
    ):
        super().__init__()
        model, _, model_transforms, _ = model_utils.load_model(name)

        self.device = device
        self.model = model.to(self.device)
        self.model_transforms = model_transforms

    def embed_vision(self, observations):
        if isinstance(observations, np.ndarray):
            observations = torch.from_numpy(observations).float()
        if len(observations.shape) != 4:
            observations = observations.unsqueeze(0)
        observations = observations.permute(0, 3, 1, 2)
        transformed_img = self.model_transforms(observations).to(self.device)
        with torch.inference_mode():
            embedding = self.model(transformed_img)
        return embedding.squeeze().detach().cpu().numpy()

    def embed_language(self, observations):
        raise NotImplementedError

    @property
    def perception_embedding_size(self) -> int:
        return self.model.config.hidden_size

    @property
    def language_embedding_size(self) -> int:
        raise NotImplementedError

class CacheGoals:
    def __init__(
        self,
        config_path: str,
        data_path: str = "",
        split: str = "train",
        output_path: str = "",
    ) -> None:
        self.device = torch.device("cuda")

        self.config_path = config_path
        self.data_path = data_path
        self.output_path = output_path
        self.split = split
        self.encoder_name="vc1"
        self.init_visual_encoder()

    def init_visual_encoder(self):
        self.encoder = VC1Encoder(device=self.device)

    def config_env(self, scene):
        config = get_config(self.config_path)
        with read_write(config):
            config.habitat.dataset.data_path = os.path.join(
                self.data_path, f"{self.split}/{self.split}.json.gz"
            )
            config.habitat.dataset.content_scenes = [scene]

        env = habitat.Env(config=config)
        return env

    def run(self, scene):
        output_file = os.path.join(
            self.output_path,
            f"{scene}_{self.encoder_name}_embedding.pkl",
        )
        if os.path.exists(output_file):
            print("Scene already cached: {}".format(scene))
            return

        data = {}
        env = self.config_env(scene)
        env.reset()
        
        os.makedirs(self.output_path, exist_ok=True)
                
        for episode in tqdm(env._dataset.episodes):
            episode_id = episode.episode_id
            img = env.task.sensor_suite.sensors["imagegoal"]._get_pointnav_episode_image_goal(episode)
            embedding = self.encoder.embed_vision(img)
            goal_metadata = dict(
                goal_position=episode.goals[0].position,
                embedding=embedding,
            )
            data[f"{scene}_{episode_id}"] = goal_metadata

        file = open(output_file, 'wb')
        pickle.dump(data, file)
        file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/tasks/instance_imagenav_stretch_hm3d.yaml",
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="",
    )
    args = parser.parse_args()

    cache = CacheGoals(
        config_path=args.config,
        data_path=args.input_path,
        split=args.split,
        output_path=args.output_path,
    )
    cache.run(args.scene)
