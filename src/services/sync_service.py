import os
import logging
from typing import List
from datetime import datetime
from typing import Optional

from api.spotify_client import SpotifyClient
from api.lyrics_client import LyricsClient
from storage.csv_tables import PlaylistsTable, SongsTable, PlaylistTracksTable, LyricsTable
from .lyrics_fetch_service import LyricsFetchService
from storage.state_store import StateStore
from search.hybrid_search import HybridSearchEngine

logger = logging.getLogger(__name__)


class SyncService:
    def __init__(self, data_dir: str = './data'):
        self.playlists = PlaylistsTable(f'{data_dir}/playlists.csv')
        self.songs = SongsTable(f'{data_dir}/songs.csv')
        self.playlist_tracks = PlaylistTracksTable(f'{data_dir}/playlist_tracks.csv')
        self.lyrics = LyricsTable(f'{data_dir}/lyrics.csv')
        self.state = StateStore(f'{data_dir}/state.json')

    def sync_all(self, spotify: SpotifyClient, lyrics_client: LyricsClient, 
                    lyrics_service: Optional[LyricsFetchService] = None, search_engine: Optional[HybridSearchEngine] = None) -> dict:
        summary = {
            'playlists_synced': 0,
            'playlists_skipped': 0,
            'songs_upserted': 0,
            'tracks_linked': 0,
            'links_removed': 0,
            'songs_deleted': 0,
            'lyrics_deleted': 0,
            'lyrics_job_started': False,
            'lyrics_enqueued': 0,
        }

        # fetch playlists
        sp_playlists = spotify.get_user_playlists()
        now_iso = datetime.utcnow().isoformat()
        prev_sync_iso = self.state.get_last_synced_at()
        prev_sync_dt = None
        if prev_sync_iso:
            try:
                prev_sync_dt = datetime.fromisoformat(prev_sync_iso+'+00:00')
            except Exception:
                prev_sync_dt = None

        for pl in sp_playlists:
            playlist_id = pl['id']
            name = pl['name']
            snapshot_id = pl['snapshot_id']
            tracks_total = pl['tracks_count']

            existing = self.playlists.get(playlist_id)
            # Skip entire playlist if snapshot_id unchanged
            if existing and existing.get('snapshot_id') == snapshot_id and prev_sync_dt:
                summary['playlists_skipped'] += 1
                continue

            # Upsert playlist row
            self.playlists.upsert({
                'playlist_id': playlist_id,
                'name': name,
                'snapshot_id': snapshot_id,
                'tracks_total': tracks_total,
                'original_id': playlist_id,
                'source': 'spotify'
            })
            summary['playlists_synced'] += 1

            # Fetch tracks for this playlist
            tracks = spotify.get_playlist_tracks(playlist_id)

            # Precompute song ids existing in songs.csv once
            existing_song_rows = self.songs._read_all()
            existing_ids = {r.get('song_id') for r in existing_song_rows if r.get('song_id')}

            def _to_dt(val):
                if not val:
                    return None
                if hasattr(val, 'tzinfo'):
                    return val
                try:
                    return datetime.fromisoformat(str(val)+'+00:00')
                except Exception:
                    return None

            present_ids: List[str] = []
            for track in tracks:
                sid = getattr(track, 'song_id', None)
                if not sid:
                    continue
                present_ids.append(sid)

                # Upsert song if missing
                if sid not in existing_ids:
                    self.songs.upsert({
                        'song_id': sid,
                        'original_id': sid,
                        'isrc_id': getattr(track, 'isrc_id', ''),
                        'title': track.title,
                        'artist': track.artist,
                        'album': track.album or '',
                        'duration_ms': track.duration_ms or '',
                        'popularity': track.popularity or '',
                        'release_date': track.release_date or '',
                        'source': 'spotify'
                    })
                    summary['songs_upserted'] += 1

                # Upsert playlist-track link only if first run or newly added
                added_at = getattr(track, 'added_at', None)
                added_at_dt = _to_dt(added_at)
                if (prev_sync_dt is None) or (added_at_dt and added_at_dt > prev_sync_dt):
                    added_at_iso = added_at.isoformat() if hasattr(added_at, 'isoformat') else (added_at or '')
                    self.playlist_tracks.upsert(playlist_id, sid, added_at_iso)
                    summary['tracks_linked'] += 1

                # Lyrics fetching moved to background job (optional)

            # Remove playlist links that are no longer present
            removed = self.playlist_tracks.remove_missing(playlist_id, present_ids)
            summary['links_removed'] += removed

        # After syncing all playlists, remove songs (and lyrics) no longer referenced
        referenced = self.playlist_tracks.all_song_ids()
        # Build set of existing song ids
        existing_song_rows = self.songs._read_all()
        existing_ids = {r.get('song_id') for r in existing_song_rows if r.get('song_id')}
        to_delete = existing_ids - referenced
        if to_delete:
            summary['songs_deleted'] = self.songs.delete_ids(to_delete)
            summary['lyrics_deleted'] = self.lyrics.delete_ids(to_delete)
            ### delete related embeddings and recompute keyword index matrices
            if summary['lyrics_deleted'] > 0 and search_engine is not None:
                for song_id in to_delete:
                    search_engine.embedding_manager.delete_song(song_id)
                search_engine.build_phrase_and_word_index()
                search_engine.build_keyword_index('tf-idf')
                search_engine.build_keyword_index('bm25')


        # Optionally trigger background lyrics fetch after sync
        try:
            auto = os.getenv('AUTO_LYRICS_AFTER_SYNC', 'true').lower() == 'true'
            if auto and lyrics_service is not None:
                res = lyrics_service.start(lyrics_client, search_engine)
                summary['lyrics_job_started'] = (res or {}).get('status') == 'started'
                summary['lyrics_enqueued'] = (res or {}).get('enqueued', 0)
        except Exception as e:
            logger.warning(f"Failed to start lyrics job after sync: {e}")

        # Update global last synced at
        self.state.set_last_synced_at(now_iso)

        return summary
