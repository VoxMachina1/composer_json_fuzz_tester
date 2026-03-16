import os
import yaml
import json
from pathlib import Path
from dotenv import load_dotenv

def load_config(config_filename="config/strategy_config.yaml"):
    """Loads the YAML configuration and validates environment variables."""
    
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
            api_keys =[k.strip() for k in keys_str.split(",") if k.strip()]
    except Exception as e:
        raise ValueError(f"Failed to parse TIINGO_API_KEYS: {e}")

    if not api_keys:
        raise ValueError("API keys list is empty.")
        
    # Load YAML configuration
    config_path = base_dir / config_filename
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")
        
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        
    return config, api_keys

# --- TEST ---
if __name__ == "__main__":
    try:
        cfg, keys = load_config()
        print("Config loaded successfully!")
        print(f"Signal Assets: {cfg['signal_assets']}")
        print(f"Number of API Keys Loaded: {len(keys)}")
        print(f"API Keys List: {keys}")
        print("PASS")
    except Exception as e:
        print(f"FAIL: {e}")