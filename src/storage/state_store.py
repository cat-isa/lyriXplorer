import os
import json
from typing import Optional


class StateStore:
    def __init__(self, path: str = './data/state.json'):
        self.path = path
        d = os.path.dirname(path)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
        if not os.path.exists(self.path):
            self._write({})

    def _read(self) -> dict:
        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                return json.load(f) or {}
        except Exception:
            return {}

    def _write(self, data: dict) -> None:
        tmp = self.path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        os.replace(tmp, self.path)

    def get_last_synced_at(self) -> Optional[str]:
        return self._read().get('last_synced_at')

    def set_last_synced_at(self, dt_iso: str) -> None:
        data = self._read()
        data['last_synced_at'] = dt_iso
        self._write(data)
