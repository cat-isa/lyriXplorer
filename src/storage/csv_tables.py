import os
from datetime import datetime
import csv
import pandas as pd
import threading
import time
from typing import Dict, Any, List, Optional, Iterable

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

    def __init__(self, path: str):
        super().__init__(path)
        self._index: Optional[Dict[str, Dict[str, Any]]] = None

    def _build_index(self):
        """Lazy index building - builds on first access."""
        if self._index is None:
            rows = self._read_all()
            self._index = {r['playlist_id']: r for r in rows if r.get('playlist_id')}

    def _invalidate_index(self):
        """Invalidate index after writes."""
        self._index = None

    def upsert(self, playlist: Dict[str, Any]) -> None:
        rows = self._read_all()
        pid = playlist.get('playlist_id')
        for r in rows:
            if r['playlist_id'] == pid:
                r.update({k: '' if v is None else v for k, v in playlist.items() if k in self.FIELDNAMES})
                self._write_all(rows)
                self._invalidate_index()
                return
        rows.append({k: '' if v is None else v for k, v in playlist.items()})
        self._write_all(rows)
        self._invalidate_index()

    def get(self, playlist_id: str) -> Optional[Dict[str, Any]]:
        self._build_index()
        return self._index.get(playlist_id)

    def all(self) -> List[Dict[str, Any]]:
        return self._read_all()


class SongsTable(_BaseCSV):
    FIELDNAMES = ['song_id', 'original_id', 'isrc_id', 'title', 'artist', 'album', 'duration_ms', 
                    'popularity', 'release_date', 'source']

    def __init__(self, path: str):
        super().__init__(path)
        self._index: Optional[Dict[str, Dict[str, Any]]] = None

    def _build_index(self):
        """Lazy index building - builds on first access."""
        if self._index is None:
            rows = self._read_all()
            self._index = {(r.get('song_id') or '').strip(): r for r in rows if (r.get('song_id') or '').strip()}

    def _invalidate_index(self):
        """Invalidate index after writes."""
        self._index = None

    def upsert(self, song: Dict[str, Any]) -> None:
        rows = self._read_all()
        sid = (song.get('song_id') or '').strip()
        for r in rows:
            if (r.get('song_id') or '').strip() == sid:
                r.update({k: '' if v is None else v for k, v in song.items() if k in self.FIELDNAMES})
                self._write_all(rows)
                self._invalidate_index()
                return
        rows.append({k: '' if v is None else v for k, v in song.items()})
        self._write_all(rows)
        self._invalidate_index()

    def get(self, song_id: str) -> Optional[Dict[str, Any]]:
        self._build_index()
        return self._index.get((song_id or '').strip())

    def delete_ids(self, ids: Iterable[str]) -> int:
        idset = set(ids)
        rows = self._read_all()
        new_rows = [r for r in rows if (r.get('song_id') or '') not in idset]
        removed = len(rows) - len(new_rows)
        if removed:
            self._write_all(new_rows)
            self._invalidate_index()
        return removed


class PlaylistTracksTable(_BaseCSV):
    FIELDNAMES = ['playlist_id', 'song_id', 'added_at']

    def __init__(self, path: str):
        super().__init__(path)
        self._index: Optional[Dict[tuple, Dict[str, Any]]] = None

    def _build_index(self):
        """Lazy index building - builds on first access with composite key."""
        if self._index is None:
            rows = self._read_all()
            self._index = {
                (r['playlist_id'], r['song_id']): r 
                for r in rows 
                if r.get('playlist_id') and r.get('song_id')
            }

    def _invalidate_index(self):
        """Invalidate index after writes."""
        self._index = None

    def upsert(self, playlist_id: str, song_id: str, added_at: Optional[str]) -> None:
        rows = self._read_all()
        key = (playlist_id, song_id)
        for r in rows:
            if r['playlist_id'] == key[0] and r['song_id'] == key[1]:
                if added_at:
                    r['added_at'] = added_at
                self._write_all(rows)
                self._invalidate_index()
                return
        rows.append({'playlist_id': playlist_id, 'song_id': song_id, 'added_at': added_at or ''})
        self._write_all(rows)
        self._invalidate_index()

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
            self._invalidate_index()
        return removed

    def list_playlist(self, playlist_id: str) -> List[Dict[str, Any]]:
        # For filtering by single field, still use scan (index is for composite key)
        return [r for r in self._read_all() if r['playlist_id'] == playlist_id]

    def all_song_ids(self) -> set:
        return {r['song_id'] for r in self._read_all() if r.get('song_id')}


class LyricsTable(_BaseCSV):
    FIELDNAMES = ['song_id', 'lyrics_text', 'language', 'source', 'last_updated']

    def __init__(self, path: str):
        super().__init__(path)
        self._index: Optional[Dict[str, Dict[str, Any]]] = None

    def _build_index(self):
        """Lazy index building - builds on first access."""
        if self._index is None:
            rows = self._read_all()
            self._index = {r['song_id']: r for r in rows if r.get('song_id')}

    def _invalidate_index(self):
        """Invalidate index after writes."""
        self._index = None

    def upsert(self, song_id: str, lyrics_text: str, language: Optional[str], source: str, last_updated: Optional[str] = None) -> None:
        rows = self._read_all()
        for r in rows:
            if r['song_id'] == song_id:
                r['lyrics_text'] = lyrics_text
                r['language'] = language or ''
                r['source'] = source
                r['last_updated'] = last_updated or datetime.utcnow().isoformat()
                self._write_all(rows)
                self._invalidate_index()
                return
        rows.append({
            'song_id': song_id,
            'lyrics_text': lyrics_text,
            'language': language or '',
            'source': source,
            'last_updated': last_updated or datetime.utcnow().isoformat(),
        })
        self._write_all(rows)
        self._invalidate_index()

    def has(self, song_id: str) -> bool:
        """Check if lyrics exist for song_id using index - O(1)."""
        self._build_index()
        idx = self._index  # Local reference for thread safety
        if idx is None:
            return False
        row = idx.get(song_id)
        return bool(row and (row.get('lyrics_text') or '').strip())

    def get_lyrics(self, song_id: str) -> str:
        """Get lyrics for song_id using index - O(1)."""
        self._build_index()
        idx = self._index
        if idx is None:
            return 'lyrics unavailable'
        row = idx.get(song_id)
        if row and (row.get('lyrics_text') or '').strip():
            return row
        return 'lyrics unavailable'

    def delete_ids(self, ids: Iterable[str]) -> int:
        idset = set(ids)
        rows = self._read_all()
        new_rows = [r for r in rows if (r.get('song_id') or '') not in idset]
        removed = len(rows) - len(new_rows)
        if removed:
            self._write_all(new_rows)
            self._invalidate_index()
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

    def __init__(self, path: str):
        super().__init__(path)
        self._index: Optional[Dict[str, Dict[str, Any]]] = None

    def _build_index(self):
        """Lazy index building - builds on first access."""
        if self._index is None:
            rows = self._read_all()
            self._index = {r['song_id']: r for r in rows if r.get('song_id')}

    def _invalidate_index(self):
        """Invalidate index after writes."""
        self._index = None

    def upsert(self, song_id: str, status: str, attempts: int, last_attempt: Optional[str], last_error: str = '') -> None:
        rows = self._read_all()
        for r in rows:
            if r.get('song_id') == song_id:
                r['status'] = status
                r['attempts'] = str(attempts)
                r['last_attempt'] = last_attempt or ''
                r['last_error'] = last_error
                self._write_all(rows)
                self._invalidate_index()
                return
        rows.append({
            'song_id': song_id,
            'status': status,
            'attempts': str(attempts),
            'last_attempt': last_attempt or '',
            'last_error': last_error,
        })
        self._write_all(rows)
        self._invalidate_index()

    def get(self, song_id: str) -> Optional[Dict[str, Any]]:
        """Get status row for song_id using index - O(1)."""
        self._build_index()
        idx = self._index
        if idx is None:
            return None
        return idx.get(song_id)

    def should_skip(self, song_id: str, cooldown_hours: int, max_attempts: int) -> bool:
        """Check if song should be skipped using indexed lookup - O(1)."""
        row = self.get(song_id)  # Now uses index
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
