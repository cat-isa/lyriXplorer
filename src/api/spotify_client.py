import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from typing import List, Dict, Any, Optional
import pandas as pd
#from ..models import Song #temp
from models import Song

import logging

logger = logging.getLogger(__name__)



class SpotifyClient:
    """Client for interacting with Spotify Web API"""
    
    def __init__(self):
        self.client_id = os.getenv("SPOTIFY_CLIENT_ID")
        self.client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
        self.redirect_uri = os.getenv("SPOTIFY_REDIRECT_URI", "http://localhost:3000/callback")
        
        if not self.client_id or not self.client_secret:
            raise ValueError("Spotify credentials not found in environment variables")
        
        self.sp = spotipy.Spotify(
            auth_manager=SpotifyOAuth(
                client_id=self.client_id,
                client_secret=self.client_secret,
                redirect_uri=self.redirect_uri,
                scope="playlist-read-private playlist-read-collaborative user-library-read"
            )
        )
    
    def get_playlist_tracks(self, playlist_id: str) -> List[Song]:
        """Fetch all tracks from a Spotify playlist"""
        try:
            tracks = []
            offset = 0
            limit = 100
            
            while True:
                results = self.sp.playlist_tracks(
                    playlist_id, 
                    offset=offset, 
                    limit=limit,
                    fields="items(track(id,name,artists,album,duration_ms,popularity,external_ids),added_at)"
                )
                
                if not results['items']:
                    break
                
                for item in results['items']:
                    track = item['track']
                    if track:  # Skip None tracks (can happen with unavailable songs)
                        song = Song(
                            song_id=track['id'],
                            original_id=track['id'],
                            isrc_id=track['external_ids']['isrc'],
                            title=track['name'],
                            artist=track['artists'][0]['name'] if track['artists'] else "Unknown Artist",
                            album=track['album']['name'] if track['album'] else None,
                            duration_ms=track['duration_ms'],
                            popularity=track['popularity'],
                            release_date=track['album']['release_date'] if track['album'] else None,
                            added_at=item['added_at'],
                            source='spotify'
                        )
                        tracks.append(song)
                
                offset += limit
                
                if len(results['items']) < limit:
                    break
            
            logger.info(f"Fetched {len(tracks)} tracks from playlist {playlist_id}")
            return tracks
            
        except Exception as e:
            logger.error(f"Error fetching playlist tracks: {e}")
            raise
    
    def get_user_playlists(self) -> List[Dict[str, Any]]:
        """Get user's playlists"""
        try:
            playlists = []
            offset = 0
            limit = 50
            
            while True:
                results = self.sp.current_user_playlists(limit=limit, offset=offset)
                
                if not results['items']:
                    break
                
                for playlist in results['items']:
                    playlists.append({
                        'id': playlist['id'],
                        'name': playlist['name'],
                        'description': playlist.get('description', ''),
                        'tracks_count': playlist['tracks']['total'],
                        'public': playlist['public'],
                        'snapshot_id': playlist['snapshot_id'] # to be used to check whether the playlist has been modified
                    })
                
                offset += limit
                
                if len(results['items']) < limit:
                    break
            
            return playlists
            
        except Exception as e:
            logger.error(f"Error fetching user playlists: {e}")
            raise
    
    def export_playlist_to_csv(self, playlist_id: str, output_path: str) -> str:
        """Export playlist tracks to CSV file"""
        tracks = self.get_playlist_tracks(playlist_id)
        
        # Convert to DataFrame
        data = []
        for track in tracks:
            data.append({
                'song_id': track.song_id,
                'original_id': track.original_id,
                'isrc_id': track.isrc_id,
                'title': track.title,
                'artist': track.artist,
                'album': track.album,
                'duration_ms': track.duration_ms,
                'popularity': track.popularity,
                'release_date': track.release_date,
                'added_at': track.added_at.isoformat(),
                'source': track.source
            })
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Exported {len(tracks)} tracks to {output_path}")
        return output_path
    
    def get_track_audio_features(self, track_id: str) -> Optional[Dict[str, Any]]:
        """Get audio features for a specific track"""
        try:
            features = self.sp.audio_features(track_id)
            return features[0] if features else None
        except Exception as e:
            logger.error(f"Error fetching audio features for track {track_id}: {e}")
            return None 