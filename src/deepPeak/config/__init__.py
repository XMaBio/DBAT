import yaml
import os
from pathlib import Path

def load_config(config_path=None):
    if config_path is None:
        config_dir = Path(__file__).parent
        config_path = config_dir / "cfg.yml"

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} not found")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    config = load_config()

    print("Config loaded successfully")