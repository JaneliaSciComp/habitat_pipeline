import hashlib
import json
import pickle
from pathlib import Path

CACHE_DIR = Path(".gui_cache")


def _make_key(animal_id, session_id, config_path, params: dict) -> str:
    payload = json.dumps(
        {"a": animal_id, "s": session_id, "c": str(config_path), **params},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def cache_path(prefix: str, animal_id, session_id, config_path, params: dict) -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    key = _make_key(animal_id, session_id, config_path, params)
    return CACHE_DIR / f"{prefix}_{key}.pkl"


def load_cache(path: Path):
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def save_cache(path: Path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)
