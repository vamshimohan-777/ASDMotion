import json
import os
import tempfile

import torch


class CheckpointManager:
    def __init__(self, root_dir="results"):
        self.root_dir = str(root_dir)
        os.makedirs(self.root_dir, exist_ok=True)

    def _atomic_json_dump(self, path, payload):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix="tmp_ckpt_", suffix=".json", dir=os.path.dirname(path) or ".")
        os.close(fd)
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def save_model(self, filename, payload):
        path = os.path.join(self.root_dir, filename)
        torch.save(payload, path)
        return path

    def save_json(self, filename, payload):
        path = os.path.join(self.root_dir, filename)
        self._atomic_json_dump(path, payload)
        return path

