import os
import json
from pathlib import Path
from dotenv import load_dotenv

def load_config(config_filename=None, config_dict=None):
    """
    Loads Tiingo API keys from the .env file.
    (YAML config loading has been removed as it is unused by the fuzzer).
    """
    base_dir = Path(__file__).resolve().parent.parent
    env_path = base_dir / ".env"
    load_dotenv(dotenv_path=env_path)

    keys_str = os.getenv("TIINGO_API_KEYS")
    if not keys_str:
        raise ValueError("TIINGO_API_KEYS not found in the environment or .env file.")

    try:
        if keys_str.strip().startswith("["):
            api_keys = json.loads(keys_str)
        else:
            api_keys = [k.strip() for k in keys_str.split(",") if k.strip()]
    except Exception as e:
        raise ValueError(f"Failed to parse TIINGO_API_KEYS: {e}")

    if not api_keys:
        raise ValueError("API keys list is empty.")

    # Return empty dict for config (unused) and the api keys
    return config_dict or {}, api_keys
