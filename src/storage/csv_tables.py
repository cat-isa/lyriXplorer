import os
from datetime import datetime
import csv
import pandas as pd
import threading
import time
from typing import Dict, Any, List, Optional, Iterable, Tuple

class _BaseCSV:
    FIELDNAMES: List[str] = []

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._ensure_dir()
        self._ensure_file()

    def _ensure_dir(self) -> None:
        d = os.path.dirname(self.path)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)

    def _ensure_file(self) -> None:
        if not os.path.exists(self.path):
            with open(self.path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                writer.writeheader()

    def _read_all(self) -> List[Dict[str, Any]]:
        with self._lock, open(self.path, 'r', newline='', encoding='utf-8') as f:
            return list(csv.DictReader(f))

    def _write_all(self, rows: List[Dict[str, Any]]) -> None:
        tmp = self.path + '.tmp'
        with self._lock:
            with open(tmp, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                writer.writeheader()
                for row in rows:
                    writer.writerow({k: '' if v is None else str(v) for k, v in row.items() if k in self.FIELDNAMES})
            # On Windows, another process/thread (e.g., AV or indexing) can momentarily lock the file.
            # Retry replace a few times with exponential backoff.
            attempts = 6
            delay = 0.02
            for i in range(attempts):
                try:
                    os.replace(tmp, self.path)
                    break
                except PermissionError:
                    if i == attempts - 1:
                        # Best effort cleanup: try to remove tmp
                        try:
                            os.remove(tmp)
                        except Exception:
                            pass
                        raise
                    time.sleep(delay)
                    delay *= 2

    def _to_dataFrame(self) -> pd.DataFrame:
        return pd.DataFrame(self._read_all())


class PlaylistsTable(_BaseCSV):
    FIELDNAMES = ['playlist_id', 'name', 'snapshot_id', 'tracks_total', 'original_id', 'source']

    def upsert(self, playlist: Dict[str, Any]) -> None:
        rows = self._read_all()
        for r in rows:
            if r['playlist_id'] == playlist.get('playlist_id'):
                r.update({k: '' if v is None else v for k, v in playlist.items() if k in self.FIELDNAMES})
                self._write_all(rows)
                return
        rows.append({k: '' if v is None else v for k, v in playlist.items()})
        self._write_all(rows)

    def get(self, playlist_id: str) -> Optional[Dict[str, Any]]:
        for r in self._read_all():
            if r['playlist_id'] == playlist_id:
                return r
        return None

    def all(self) -> List[Dict[str, Any]]:
        return self._read_all()


class SongsTable(_BaseCSV):
    FIELDNAMES = ['song_id', 'original_id', 'isrc_id', 'title', 'artist', 'album', 'duration_ms', 
                    'popularity', 'release_date', 'source']

    def upsert(self, song: Dict[str, Any]) -> None:
        rows = self._read_all()
        sid = (song.get('song_id') or '').strip()
        for r in rows:
            if (r.get('song_id') or '').strip() == sid:
                r.update({k: '' if v is None else v for k, v in song.items() if k in self.FIELDNAMES})
                self._write_all(rows)
                return
        rows.append({k: '' if v is None else v for k, v in song.items()})
        self._write_all(rows)

    def get(self, song_id: str) -> Optional[Dict[str, Any]]:
        for r in self._read_all():
            if r.get('song_id') == song_id:
                return r
        return None

    def delete_ids(self, ids: Iterable[str]) -> int:
        idset = set(ids)
        rows = self._read_all()
        new_rows = [r for r in rows if (r.get('song_id') or '') not in idset]
        removed = len(rows) - len(new_rows)
        if removed:
            self._write_all(new_rows)
        return removed


class PlaylistTracksTable(_BaseCSV):
    FIELDNAMES = ['playlist_id', 'song_id', 'added_at']

    def upsert(self, playlist_id: str, song_id: str, added_at: Optional[str]) -> None:
        rows = self._read_all()
        key = (playlist_id, song_id)
        for r in rows:
            if r['playlist_id'] == key[0] and r['song_id'] == key[1]:
                if added_at:
                    r['added_at'] = added_at
                self._write_all(rows)
                return
        rows.append({'playlist_id': playlist_id, 'song_id': song_id, 'added_at': added_at or ''})
        self._write_all(rows)

    def remove_missing(self, playlist_id: str, present_song_ids: Iterable[str]) -> int:
        present = set(present_song_ids)
        rows = self._read_all()
        new_rows: List[Dict[str, Any]] = []
        removed = 0
        for r in rows:
            if r['playlist_id'] == playlist_id and r['song_id'] not in present:
                removed += 1
                continue
            new_rows.append(r)
        if removed:
            self._write_all(new_rows)
        return removed

    def list_playlist(self, playlist_id: str) -> List[Dict[str, Any]]:
        return [r for r in self._read_all() if r['playlist_id'] == playlist_id]

    def all_song_ids(self) -> set:
        return {r['song_id'] for r in self._read_all() if r.get('song_id')}


class LyricsTable(_BaseCSV):
    FIELDNAMES = ['song_id', 'lyrics_text', 'language', 'source', 'last_updated']

    def upsert(self, song_id: str, lyrics_text: str, language: Optional[str], source: str, last_updated: Optional[str] = None) -> None:
        rows = self._read_all()
        for r in rows:
            if r['song_id'] == song_id:
                r['lyrics_text'] = lyrics_text
                r['language'] = language or ''
                r['source'] = source
                r['last_updated'] = last_updated or datetime.utcnow().isoformat()
                self._write_all(rows)
                return
        rows.append({
            'song_id': song_id,
            'lyrics_text': lyrics_text,
            'language': language or '',
            'source': source,
            'last_updated': last_updated or datetime.utcnow().isoformat(),
        })
        self._write_all(rows)

    def has(self, song_id: str) -> bool:
        for r in self._read_all():
            if r['song_id'] == song_id and (r.get('lyrics_text') or '').strip():
                return True
        return False

    def get_lyrics(self, song_id: str) -> str:
        for r in self._read_all():
            if r['song_id'] == song_id and (r.get('lyrics_text') or '').strip():
                return r.get('lyrics_text', '')
        return 'lyrics unavailable'

    def delete_ids(self, ids: Iterable[str]) -> int:
        idset = set(ids)
        rows = self._read_all()
        new_rows = [r for r in rows if (r.get('song_id') or '') not in idset]
        removed = len(rows) - len(new_rows)
        if removed:
            self._write_all(new_rows)
        return removed


class LyricsStatusTable(_BaseCSV):
    """Tracks fetch status for lyrics per song_id.

    Fields:
      - song_id
      - status: pending|success|failed
      - attempts: int
      - last_attempt: iso datetime
      - last_error: str
    """

    FIELDNAMES = ['song_id', 'status', 'attempts', 'last_attempt', 'last_error']

    def upsert(self, song_id: str, status: str, attempts: int, last_attempt: Optional[str], last_error: str = '') -> None:
        rows = self._read_all()
        for r in rows:
            if r.get('song_id') == song_id:
                r['status'] = status
                r['attempts'] = str(attempts)
                r['last_attempt'] = last_attempt or ''
                r['last_error'] = last_error
                self._write_all(rows)
                return
        rows.append({
            'song_id': song_id,
            'status': status,
            'attempts': str(attempts),
            'last_attempt': last_attempt or '',
            'last_error': last_error,
        })
        self._write_all(rows)

    def get(self, song_id: str) -> Optional[Dict[str, Any]]:
        for r in self._read_all():
            if r.get('song_id') == song_id:
                return r
        return None

    def should_skip(self, song_id: str, cooldown_hours: int, max_attempts: int) -> bool:
        row = self.get(song_id)
        if not row:
            return False
        try:
            attempts = int(row.get('attempts') or '0')
        except Exception:
            attempts = 0
        if attempts >= max_attempts:
            return True
        last_attempt = row.get('last_attempt') or ''
        if not last_attempt:
            return False
        try:
            dt = datetime.fromisoformat(last_attempt.replace('Z', '+00:00'))
        except Exception:
            return False
        delta = datetime.utcnow() - dt
        return delta.total_seconds() < cooldown_hours * 3600
