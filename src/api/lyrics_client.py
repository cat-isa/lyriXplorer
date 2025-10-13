import requests
import time
import re
import logging
from typing import Optional, Dict, Any
#from ..models import Lyrics - temp
from models import Lyrics

logger = logging.getLogger(__name__)


class LyricsClient:
    """Client for fetching lyrics using LRCLib."""

    BASE_URL = "https://lrclib.net/api"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'LyriXplorer/1.0 (+https://github.com/yourusername/lyrixplorer)'
        })

    def _lrclib_search(
        self,
        track_name: Optional[str] = None,
        artist_name: Optional[str] = None,
        album_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        try:
            params: Dict[str, Any] = {
                'track_name': track_name or '',
                'artist_name': artist_name or '',
                'album_name': album_name or '',
            }
            resp = self.session.get(f"{self.BASE_URL}/search", params=params, timeout=20)
            resp.raise_for_status()
            results = resp.json() or []
            if not isinstance(results, list) or len(results) == 0:
                return None
            return results[0]
        except Exception as e:
            logger.warning(f"LRCLib search failed: {e}")
            return None

    def _lrclib_get_lyrics(
        self,
        track_name: Optional[str] = None,
        artist_name: Optional[str] = None,
        album_name: Optional[str] = None,
    ) -> Optional[str]:
        match = self._lrclib_search(track_name=track_name, artist_name=artist_name, album_name=album_name)
        if not match:
            return None

        plain = (match.get('plainLyrics') or '').strip()
        if plain:
            return plain
        synced = (match.get('syncedLyrics') or '').strip()
        if not synced:
            return None
        # Strip timestamps like [mm:ss.xx]
        synced_plain = re.sub(r"\[\d{1,2}:\d{2}(?:\.\d{1,2})?\]\s*", "", synced)
        return synced_plain or None

    def get_lyrics(
        self,
        song_title: str,
        artist: str,
        album: Optional[str] = None,
        duration_ms: Optional[int] = None,
        isrc: Optional[str] = None,
    ) -> Optional[Lyrics]:
        """Fetch lyrics via LRCLib and return a Lyrics model."""
        # duration and isrc are currently unused by LRCLib; kept for future use
        lyrics_text = self._lrclib_get_lyrics(song_title, artist, album)
        if lyrics_text:
            return Lyrics(
                song_id=f"{song_title}_{artist}".replace(" ", "_").lower(),
                lyrics_text=lyrics_text,
                source="lrclib",
            )
        logger.warning(f"Could not fetch lyrics for: {song_title} - {artist}")
        return None

    def batch_get_lyrics(self, songs: list, delay: float = 1.0) -> Dict[str, Lyrics]:
        """Fetch lyrics for multiple songs with rate limiting."""
        results: Dict[str, Lyrics] = {}
        for i, song in enumerate(songs):
            logger.info(f"Fetching lyrics for {i+1}/{len(songs)}: {song.title} - {song.artist}")
            album = getattr(song, 'album', None)
            lyrics = self.get_lyrics(song.title, song.artist, album=album)
            if lyrics:
                key = getattr(song, 'song_id', None) or f"{song.title}_{song.artist}"
                results[key] = lyrics
            if i < len(songs) - 1:
                time.sleep(delay)
        logger.info(f"Successfully fetched lyrics for {len(results)} out of {len(songs)} songs")
        return results