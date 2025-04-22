import json
import os
import fcntl
from pathlib import Path


class TokenTracker:
    def __init__(self, initial_tokens=1000000):
        """Initialize token tracker with default values"""
        self.data_dir = Path(__file__).resolve().parent.parent / "data"
        self.data_dir.mkdir(exist_ok=True)
        self.token_file = self.data_dir / "token_usage.json"
        self.initial_tokens = initial_tokens

    def update_usage(self, api_key, model_name, tokens):
        """Atomic update of token usage with exclusive file locking"""
        supported_models = ["qvq-max", "qvq-max-latest", "qvq-max-2025-03-25"]

        with open(self.token_file, "r+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)  # Exclusive lock for atomic write

            try:
                f.seek(0)
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}

            # Initialize API key if missing
            data.setdefault(api_key, {})

            # Initialize model entry if missing
            model_data = data[api_key]
            for model in supported_models:
                model_data.setdefault(model, self.initial_tokens)

            # Validate model name
            if model_name not in supported_models:
                raise ValueError(
                    f"Invalid model name: {model_name}. Supported models: {supported_models}"
                )

            # Update tokens (prevent negative values)
            current_tokens = model_data[model_name]
            new_tokens = max(0, current_tokens - tokens)
            model_data[model_name] = new_tokens

            # Write back to file
            f.seek(0)
            f.truncate()
            json.dump(data, f, indent=2)
            f.flush()

            fcntl.flock(f, fcntl.LOCK_UN)  # Release lock

    def get_usage(self, api_key, model_name):
        """Thread-safe retrieval of token balance"""
        supported_models = ["qvq-max", "qvq-max-latest", "qvq-max-2025-03-25"]

        with open(self.token_file, "r") as f:
            fcntl.flock(f, fcntl.LOCK_SH)  # Shared lock for safe read
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
            fcntl.flock(f, fcntl.LOCK_UN)  # Release lock early

        # Validate inputs
        if model_name not in supported_models:
            raise ValueError(f"Invalid model name: {model_name}")

        # Calculate current tokens
        api_data = data.get(api_key, {})
        model_tokens = api_data.get(model_name, self.initial_tokens)
        return max(0, model_tokens)  # Prevent negative values
