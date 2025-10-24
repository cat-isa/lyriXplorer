"""
ColBERT adapter to integrate with existing EmbeddingManager interface
This allows minimal changes to the rest of the codebase
"""

from typing import List, Dict, Any, Optional
from .colbert_manager import ColBERTManager
from models import Song, Lyrics, SongWithLyrics
import logging

logger = logging.getLogger(__name__)


class ColBERTEmbeddingManager:
    """
    Adapter that makes ColBERTManager compatible with existing EmbeddingManager interface
    """
    
    def __init__(self, persist_directory: str = "./stored_embeddings"):
        self.colbert_manager = ColBERTManager(persist_directory)
        
        # Expose the same interface as the original EmbeddingManager
        self.collection = self.colbert_manager.collection
        self.client = self.colbert_manager.client
    
    def generate_embedding(self, text: str, prompt_name: Optional[str] = None, batch_size: Optional[int] = 1) -> List[float]:
        """Generate embedding for a given text (compatibility method)"""
        try:
            token_embeddings, attention_mask = self.colbert_manager._encode_text(text)
            # Return flattened embeddings for compatibility
            return token_embeddings.flatten().tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def add_song_with_lyrics(self, song: Song, lyrics: Lyrics) -> bool:
        """Add a song with its lyrics to the vector database"""
        return self.colbert_manager.add_song_with_lyrics(song, lyrics)
    
    def add_batch(self, songs_with_lyrics: List[Dict[str, Any]]) -> bool:
        """Add multiple songs with their lyrics to the vector database"""
        try:
            successful_adds = 0
            for song_data in songs_with_lyrics:
                # Create Song and Lyrics objects
                song = Song(
                    song_id=song_data['song_id'],
                    original_id=song_data.get('original_id', ''),
                    title=song_data['title'],
                    artist=song_data['artist'],
                    album=song_data.get('album'),
                    duration_ms=song_data.get('duration_ms'),
                    popularity=song_data.get('popularity'),
                    release_date=song_data.get('release_date'),
                    source=song_data.get('source', 'spotify')
                )
                
                lyrics = Lyrics(
                    song_id=song_data['song_id'],
                    lyrics_text=song_data['lyrics_text'],
                    source=song_data.get('lyrics_source', 'lrclib')
                )
                
                if self.add_song_with_lyrics(song, lyrics):
                    successful_adds += 1
            
            logger.info(f"Added {successful_adds} songs to ColBERT database")
            return successful_adds > 0
            
        except Exception as e:
            logger.error(f"Error adding batch to ColBERT database: {e}")
            return False
    
    def batch_add_songs(self, songs_with_lyrics: List[SongWithLyrics]) -> int:
        """Add multiple songs with lyrics to the vector database"""
        successful_adds = 0
        
        for song_with_lyrics in songs_with_lyrics:
            if song_with_lyrics.lyrics:
                if self.add_song_with_lyrics(song_with_lyrics.song, song_with_lyrics.lyrics):
                    successful_adds += 1
        
        logger.info(f"Successfully added {successful_adds} out of {len(songs_with_lyrics)} songs")
        return successful_adds
    
    def search_similar(self, query: str, n_results: int = 10, 
                      filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for songs with similar lyrics"""
        return self.colbert_manager.search_songs(query, n_results, filters)
    
    def search_verse_similar(self, query: str, n_results: int = 10, 
                            filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for verses with similar content"""
        return self.colbert_manager.search_verses(query, n_results, filters)
    
    def hybrid_verse_search(self, query: str, n_results: int = 10, 
                           semantic_weight: float = 0.7,
                           filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Hybrid search combining verse-level and song-level results"""
        return self.colbert_manager.hybrid_search(query, n_results, semantic_weight, filters)
    
    def get_verse_context(self, verse_id: str, context_radius: int = 1) -> Optional[str]:
        """Get context around a specific verse"""
        try:
            # Get the verse
            verse_result = self.collection.get(
                ids=[verse_id],
                include=["metadatas", "documents"]
            )
            
            if not verse_result['metadatas'] or not verse_result['metadatas'][0]:
                return None
            
            metadata = verse_result['metadatas'][0][0]
            song_id = metadata['song_id']
            
            # Get all verses for this song
            song_verses = self.collection.get(
                where={"song_id": song_id, "doc_type": "verse"},
                include=["metadatas", "documents"]
            )
            
            if not song_verses['metadatas']:
                return None
            
            # Sort by verse index
            verse_data = list(zip(song_verses['metadatas'], song_verses['documents']))
            verse_data.sort(key=lambda x: int(x[0].get('verse_index', 0)))
            
            # Find the target verse position
            target_position = None
            for i, (meta, doc) in enumerate(verse_data):
                if meta['verse_id'] == verse_id:
                    target_position = i
                    break
            
            if target_position is None:
                return None
            
            # Get context
            start_idx = max(0, target_position - context_radius)
            end_idx = min(len(verse_data), target_position + context_radius + 1)
            
            context_verses = verse_data[start_idx:end_idx]
            context_text = "\n".join([doc for _, doc in context_verses])
            
            return context_text
            
        except Exception as e:
            logger.error(f"Error getting verse context: {e}")
            return None
    
    def add_verse_embeddings(self, song: Song, lyrics: Lyrics, include_overlapping: bool = True) -> int:
        """Add verse-level embeddings for a song"""
        return self.colbert_manager.add_verse_embeddings(song, lyrics, include_overlapping)
    
    def delete_verse_embeddings(self, song_id: str) -> bool:
        """Delete all verse embeddings for a song"""
        try:
            # Delete all verses for this song
            verse_results = self.collection.get(
                where={"song_id": song_id, "doc_type": {"$in": ["verse", "verse_pair"]}},
                include=["ids"]
            )
            
            if verse_results['ids']:
                self.collection.delete(ids=verse_results['ids'])
                logger.info(f"Deleted verse embeddings for song: {song_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error deleting verse embeddings: {e}")
            return False
    
    def get_song_embedding(self, song_id: str) -> Optional[List[float]]:
        """Get embedding for a specific song (compatibility method)"""
        try:
            results = self.collection.get(
                ids=[song_id],
                include=["embeddings"]
            )
            
            if results['embeddings'] and results['embeddings'][0]:
                return results['embeddings'][0]
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting song embedding: {e}")
            return None
    
    def update_song_lyrics(self, song_id: str, new_lyrics: Lyrics) -> bool:
        """Update lyrics and re-generate embedding for a song"""
        try:
            # Delete old embeddings
            self.delete_song(song_id)
            
            # Get song data and recreate
            from storage.csv_tables import SongsTable
            songs_table = SongsTable('./data/songs.csv')
            song_data = songs_table.get(song_id)
            
            if song_data:
                song = Song(**song_data)
                return self.add_song_with_lyrics(song, new_lyrics)
            
            return False
            
        except Exception as e:
            logger.error(f"Error updating song lyrics: {e}")
            return False
    
    def delete_song(self, song_id: str) -> bool:
        """Delete a song from the vector database"""
        return self.colbert_manager.delete_song(song_id)
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector database"""
        return self.colbert_manager.get_database_stats()
