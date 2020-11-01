import base64
from pathlib import Path
import yaml
import torch


def load_settings(root_path):
    with open(root_path / "settings.yaml") as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        return yaml.load(file, Loader=yaml.FullLoader)


def load_model(model_string):
    # Write to temp file for kaggle submission
    with open("model.dat", "wb") as f:
        f.write(base64.b64decode(model_string))
        f.close()
    return torch.load('model.dat')


ROOT_PATH = Path(__file__).resolve().parent.parent
SETTINGS = load_settings(ROOT_PATH)

_model_path = ROOT_PATH / SETTINGS["learn"]["models"]["save_dir"]
_robot_agent_model_path = _model_path / SETTINGS["learn"]["models"]["robot_agent_file"]

ROBOT_AGENT_STATE_DICT = torch.load(_robot_agent_model_path)


TORCH_DEVICE = torch.device("cuda")
