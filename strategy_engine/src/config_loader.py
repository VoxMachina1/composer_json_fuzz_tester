import os
import yaml
import json
from pathlib import Path
from dotenv import load_dotenv

def load_config(config_filename="config/strategy_config.yaml", config_dict=None):
    """
    Loads configuration and validates environment variables.

    Can be called in two ways:
      1. load_config("config/my_config.yaml")
         Loads from a YAML file as before — no change to existing behaviour.
      2. load_config(config_dict={...})
         Accepts a pre-built config dict directly, skipping file I/O entirely.
         Used by run_analysis.py to pass programmatically generated configs.

    In both cases, API keys are still loaded from the .env file.
    """

    # Robustly find the `strategy_engine` root directory
    # __file__ is src/config_loader.py, parent is src/, parent.parent is strategy_engine/
    base_dir = Path(__file__).resolve().parent.parent

    # Load environment variables from the .env file in the root
    env_path = base_dir / ".env"
    load_dotenv(dotenv_path=env_path)

    # Securely retrieve the API keys
    keys_str = os.getenv("TIINGO_API_KEYS")
    if not keys_str:
        raise ValueError("TIINGO_API_KEYS not found in the environment or .env file.")

    # Parse the keys (supports both comma-separated and JSON array formats)
    try:
        if keys_str.strip().startswith("["):
            api_keys = json.loads(keys_str)
        else:
            api_keys = [k.strip() for k in keys_str.split(",") if k.strip()]
    except Exception as e:
        raise ValueError(f"Failed to parse TIINGO_API_KEYS: {e}")

    if not api_keys:
        raise ValueError("API keys list is empty.")

    # If a dict was passed directly, use it — skip file loading entirely
    if config_dict is not None:
        return config_dict, api_keys

    # Otherwise load from YAML file as normal
    config_path = base_dir / config_filename
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config, api_keys


# --- TEST ---
if __name__ == "__main__":
    try:
        # Test 1: normal file load
        cfg, keys = load_config()
        print("Config loaded from file successfully!")
        print(f"Signal Assets: {cfg['signal_assets']}")
        print(f"Number of API Keys Loaded: {len(keys)}")

        # Test 2: dict passthrough
        dummy = {"signal_assets": ["QQQ"], "target_assets": ["TQQQ"], "test": True}
        cfg2, keys2 = load_config(config_dict=dummy)
        assert cfg2["test"] is True, "Dict passthrough failed"
        assert len(keys2) > 0, "Keys missing in dict mode"
        print("\nDict passthrough test passed!")
        print("PASS")
    except Exception as e:
        print(f"FAIL: {e}")