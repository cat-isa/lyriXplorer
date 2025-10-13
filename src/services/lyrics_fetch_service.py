import os
import time
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging

#from ..api.lyrics_client import LyricsClient
from api.lyrics_client import LyricsClient
#from ..storage.csv_tables import SongsTable, LyricsTable, LyricsStatusTable
from storage.csv_tables import SongsTable, LyricsTable, LyricsStatusTable
from search.hybrid_search import HybridSearchEngine

logger = logging.getLogger(__name__)


class _RateLimiter:
    def __init__(self, rps: float):
        self.rps = max(0.1, float(rps))
        self.interval = 1.0 / self.rps
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self):
        with self._lock:
            now = time.time()
            delay = self.interval - (now - self._last)
            if delay > 0:
                time.sleep(delay)
            self._last = time.time()


class LyricsFetchService:
    def __init__(self, data_dir: str = './data'):
        # Tables
        self.songs = SongsTable(f'{data_dir}/songs.csv')
        self.lyrics = LyricsTable(f'{data_dir}/lyrics.csv')
        self.lyrics_status = LyricsStatusTable(f'{data_dir}/lyrics_status.csv')

        # Settings
        self.max_concurrency = int(os.getenv('LYRICS_MAX_CONCURRENCY', '2'))
        self.rps = float(os.getenv('LYRICS_RPS', '1'))  # requests per second global
        self.batch_size = int(os.getenv('LYRICS_BATCH_SIZE', '10'))
        self.max_per_run = int(os.getenv('LYRICS_MAX_PER_RUN', '200'))
        self.max_attempts = int(os.getenv('LYRICS_MAX_ATTEMPTS', '3'))
        self.cooldown_hours = int(os.getenv('LYRICS_COOLDOWN_HOURS', '72'))

        # Runtime state
        self._queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._workers: List[threading.Thread] = []
        self._executor: Optional[ThreadPoolExecutor] = None
        self._lock = threading.Lock()
        self._rate_limiter = _RateLimiter(self.rps)

        self._running = False
        self._paused = False
        self._cancelled = False

        self._enqueued = 0
        self._processed = 0
        self._succeeded = 0
        self._failed = 0
        self._started_at: Optional[str] = None
        self._last_update: Optional[str] = None

        # Batch buffer shared among workers guarded by lock
        self._batch_buffer: List[Dict[str, Any]] = []
        self._refilling = False

    def _enqueue_missing(self):
        # Build list of songs missing lyrics
        rows = self.songs._read_all()
        enqueued = 0
        for r in rows:
            sid = r.get('song_id')
            if not sid or self.lyrics.has(sid):
                continue
            if self.lyrics_status.should_skip(sid, self.cooldown_hours, self.max_attempts):
                continue

            item = {
                'song_id': sid,
                'title': r.get('title') or '',
                'artist': r.get('artist') or '',
                'album': r.get('album') or '', 
                'duration_ms': str(r.get('duration_ms') or 0),
                'popularity': str(r.get('popularity') or 0),
                'release_date': r.get('release_date') or '', 
            }
            self._queue.put(item)
            enqueued += 1
            if enqueued >= self.max_per_run:
                break
        self._enqueued = enqueued

    def start(self, lyrics_client: LyricsClient, search_engine: Optional[HybridSearchEngine] = None) -> Dict[str, Any]: 
        with self._lock:
            if self._running:
                return {'status': 'already_running'}
            # Reset state
            self._paused = False
            self._cancelled = False
            self._processed = 0
            self._succeeded = 0
            self._failed = 0
            self._started_at = datetime.utcnow().isoformat()
            self._last_update = self._started_at
            self._refilling = False
            # Clear queue
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except Exception:
                    break
            # Enqueue missing
            self._enqueue_missing()
            self._running = True
            self._executor = ThreadPoolExecutor(max_workers=self.max_concurrency)
            self._workers = []
            for _ in range(self.max_concurrency):
                t = threading.Thread(target=self._worker_loop, args=(lyrics_client,search_engine,), daemon=True)
                t.start()
                self._workers.append(t)
        return {'status': 'started', 'enqueued': self._enqueued}

    def pause(self):
        with self._lock:
            if not self._running:
                return {'status': 'not_running'}
            self._paused = True
            return {'status': 'paused'}

    def resume(self):
        with self._lock:
            if not self._running:
                return {'status': 'not_running'}
            self._paused = False
            return {'status': 'resumed'}

    def cancel(self):
        with self._lock:
            if not self._running:
                return {'status': 'not_running'}
            self._cancelled = True
            self._paused = False
        # Wait for workers to exit
        for t in list(self._workers):
            t.join(timeout=0.1)
        with self._lock:
            self._running = False
            self._workers = []
            self._executor = None
        # Flush any remaining batch buffer
        self._flush_batch(force=True)
        return {'status': 'cancelled'}

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                'running': self._running,
                'paused': self._paused,
                'cancelled': self._cancelled,
                'enqueued': self._enqueued,
                'processed': self._processed,
                'succeeded': self._succeeded,
                'failed': self._failed,
                'started_at': self._started_at,
                'last_update': self._last_update,
                'queue_size': self._queue.qsize(),
                'settings': {
                    'max_concurrency': self.max_concurrency,
                    'rps': self.rps,
                    'batch_size': self.batch_size,
                    'max_per_run': self.max_per_run,
                }
            }

    def _worker_loop(self, lyrics_client: LyricsClient, search_engine: Optional[HybridSearchEngine] = None):
        while True:
            # Check cancel
            with self._lock:
                if self._cancelled:
                    break
                paused = self._paused
                running = self._running
            if not running:
                break
            if paused:
                time.sleep(0.2)
                continue

            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                # Attempt to refill once if possible (only one worker performs refill)
                should_refill = False
                with self._lock:
                    if self._running and not self._cancelled and not self._refilling:
                        self._refilling = True
                        should_refill = True
                if should_refill:
                    try:
                        prev_enq = self._enqueued
                        self._enqueue_missing()
                        # If nothing new was enqueued, stop service
                        if self._queue.empty() and self._enqueued == prev_enq:
                            with self._lock:
                                self._running = False
                                self._refilling = False
                                print(f"{self._succeeded} new lyrics added.") ###TEMP: print in log instead
                                break
                    finally:
                        with self._lock:
                            self._refilling = False
                else:
                    # brief backoff and retry
                    time.sleep(0.2)
                continue

            song_id = item['song_id']
            title = item['title']
            artist = item['artist']
            album = item['album']
            duration_ms = item['duration_ms']
            popularity = item['popularity']
            release_date = item['release_date']

            # Global rate limit across workers
            self._rate_limiter.wait()

            try:
                lobj = lyrics_client.get_lyrics(title, artist, album=album)
                success = bool(lobj and (lobj.lyrics_text or '').strip())
                if success:
                    with self._lock:
                        self._batch_buffer.append({
                            'song_id': song_id,
                            'title': title,
                            'artist': artist,
                            'album': album,
                            'duration_ms': duration_ms,
                            'popularity': popularity,
                            'release_date': release_date,
                            'lyrics_text': lobj.lyrics_text,
                            'language': lobj.language,
                            'lyrics_source': lobj.source,
                            'lyrics_length': str(len(lobj.lyrics_text))
                        })
                        self._succeeded += 1
                        self._processed += 1
                        self._last_update = datetime.utcnow().isoformat()
                        if len(self._batch_buffer) >= self.batch_size:
                            # Flush while already holding the lock
                            self._flush_batch(force=False, already_locked=True, search_engine=search_engine)
                else:
                    with self._lock:
                        self._failed += 1
                        self._processed += 1
                        self._last_update = datetime.utcnow().isoformat()
                # update status table attempts and state
                try:
                    prev = self.lyrics_status.get(song_id)
                    attempts = int(prev.get('attempts') or '0') + 1 if prev else 1
                except Exception:
                    attempts = 1
                self.lyrics_status.upsert(
                    song_id, 'success' if success else 'failed', attempts, datetime.utcnow().isoformat(), '' if success else 'not_found'
                )
            except Exception as e:
                logger.warning(f"Lyrics fetch failed for {title} - {artist}: {e}")
                with self._lock:
                    self._failed += 1
                    self._processed += 1
                    self._last_update = datetime.utcnow().isoformat()
                # record error with attempts
                try:
                    prev = self.lyrics_status.get(song_id)
                    attempts = int(prev.get('attempts') or '0') + 1 if prev else 1
                except Exception:
                    attempts = 1
                self.lyrics_status.upsert(
                    song_id, 'failed', attempts, datetime.utcnow().isoformat(), str(e)[:200]
                )
            finally:
                self._queue.task_done()

        # Final flush when worker exits
        self._flush_batch(force=True, search_engine=search_engine)

    def _flush_batch(self, force: bool, already_locked: bool = False, search_engine: Optional[HybridSearchEngine] = None): 
        if already_locked:
            # Caller holds self._lock
            if not self._batch_buffer:
                return
            batch = self._batch_buffer[:]
            if not force and len(batch) < self.batch_size:
                return
            self._batch_buffer.clear()
        else:
            with self._lock:
                if not self._batch_buffer:
                    return
                batch = self._batch_buffer[:]
                if not force and len(batch) < self.batch_size:
                    return
                self._batch_buffer.clear()
        # Write outside lock
        for row in batch:
            try:
                self.lyrics.upsert(
                    row['song_id'], row['lyrics_text'], row.get('language'), row.get('lyrics_source') or 'lrclib'
                )
            except Exception as e:
                logger.warning(f"Batch write failed for {row.get('song_id')}: {e}")

        ### if embedding_manager in not None, call method add_batch with "batch" as an argument
        ### and recompute keyword indexes
        ### IMPORTANT: check whether we can lock these operations
        if search_engine is not None:
            search_engine.embedding_manager.add_batch(batch)
            search_engine.build_phrase_and_word_index()
            search_engine.build_keyword_index('tf-idf')
            search_engine.build_keyword_index('bm25')
