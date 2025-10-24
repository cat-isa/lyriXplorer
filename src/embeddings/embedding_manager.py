import os
import json
import pickle
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import logging
from models import Song, Lyrics, SongWithLyrics

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages vector embeddings for lyrics using sentence-transformers and ChromaDB"""
    
    def __init__(self, persist_directory: str = "./stored_embeddings"):
        self.persist_directory = persist_directory
        self.model_name = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
        
        # Initialize the embedding model
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="lyrics_embeddings",
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"Embedding manager initialized with {self.collection.count()} existing embeddings")
    
    def generate_embedding(self, text: str, prompt_name: Optional[str], batch_size: Optional[int]=1) -> List[float]:
        """Generate embedding for a given text"""
        try:
            if prompt_name:
                embedding = self.model.encode(text, convert_to_tensor=False, prompt_name=prompt_name, batch_size=batch_size)
            else:
                embedding = self.model.encode(text, convert_to_tensor=False, batch_size=batch_size)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def add_song_with_lyrics(self, song: Song, lyrics: Lyrics) -> bool:
        """Add a song with its lyrics to the vector database"""
        try:
            # Generate embedding for lyrics
            embedding = self.generate_embedding(lyrics.lyrics_text, prompt_name='document')
            
            # Create metadata
            metadata = {
                "song_id": song.song_id,
                "title": song.title,
                "artist": song.artist,
                "album": song.album or "",
                "duration_ms": str(song.duration_ms or 0),
                "popularity": str(song.popularity or 0),
                "release_date": song.release_date or "",
                "lyrics_source": lyrics.source,
                "lyrics_length": str(len(lyrics.lyrics_text))
            }
            
            # Add to ChromaDB
            self.collection.add(
                embeddings=[embedding],
                documents=[lyrics.lyrics_text],
                metadatas=[metadata],
                ids=[song.song_id]
            )
            
            logger.info(f"Added embedding for song: {song.title} - {song.artist}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding song to vector database: {e}")
            return False

    def add_batch(self, songs_with_lyrics: List[Dict[str, Any]]) -> bool:
        """Add multiple songs with their lyrics to the vector database"""
        try:
            # Generate embedding for lyrics
            batch_size = int(os.getenv('EMBEDDING_BATCH_SIZE', 32))
            all_lyrics = [s.get('lyrics_text') for s in songs_with_lyrics]
            embeddings = self.generate_embedding(all_lyrics, prompt_name='document', batch_size=batch_size)
            
            # Create metadata
            all_metadata = [ 
            {
                "song_id": s.get('song_id'),
                "title": s.get('title'),
                "artist": s.get('artist'),
                "album": s.get('album') or "",
                "duration_ms": str(s.get('duration_ms') or 0),
                "popularity": str(s.get('popularity') or 0),
                "release_date": s.get('release_date') or "",
                "lyrics_source": s.get('lyrics_source'),
                "lyrics_length": s.get('lyrics_length')
            } for s in songs_with_lyrics
            ]

            # get song_ids into a list
            song_ids = [s.get('song_id') for s in songs_with_lyrics]
            
            # Add to ChromaDB
            self.collection.add(
                embeddings=embeddings,
                documents=all_lyrics,
                metadatas=all_metadata,
                ids=song_ids
            )
            
            logger.info(f"Added embedding for {len(all_lyrics)} songs")
            return True
            
        except Exception as e:
            logger.error(f"Error adding songs to vector database: {e}")
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
        try:
            # Generate embedding for query
            query_embedding = self.generate_embedding(query, prompt_name='query')
            
            # Prepare where clause for filters
            where_clause = None
            if filters:
                where_clause = {}
                for key, value in filters.items():
                    if value is not None:
                        where_clause[key] = value
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_clause,
                include=["metadatas", "documents", "distances"]
            )
            
            # Format results
            formatted_results = []
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    formatted_results.append({
                        'song_id': results['metadatas'][0][i]['song_id'],
                        'title': results['metadatas'][0][i]['title'],
                        'artist': results['metadatas'][0][i]['artist'],
                        'album': results['metadatas'][0][i]['album'],
                        'lyrics_excerpt': results['documents'][0][i][:500] + "..." if len(results['documents'][0][i]) > 500 else results['documents'][0][i],
                        'similarity_score': 1 - results['distances'][0][i],  # Convert distance to similarity
                        'metadata': results['metadatas'][0][i]
                    })
            
            logger.info(f"Found {len(formatted_results)} similar songs for query: {query}")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching similar songs: {e}")
            return []
    
    def get_song_embedding(self, song_id: str) -> Optional[List[float]]:
        """Get embedding for a specific song"""
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
            # Generate new embedding
            new_embedding = self.generate_embedding(new_lyrics.lyrics_text)
            
            # Update metadata
            metadata = {
                "lyrics_source": new_lyrics.source,
                "lyrics_length": str(len(new_lyrics.lyrics_text))
            }
            
            # Update in ChromaDB
            self.collection.update(
                ids=[song_id],
                embeddings=[new_embedding],
                documents=[new_lyrics.lyrics_text],
                metadatas=[metadata]
            )
            
            logger.info(f"Updated lyrics and embedding for song: {song_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating song lyrics: {e}")
            return False
    
    def delete_song(self, song_id: str) -> bool:
        """Delete a song from the vector database"""
        try:
            self.collection.delete(ids=[song_id])
            logger.info(f"Deleted song from vector database: {song_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting song: {e}")
            return False
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector database"""
        try:
            count = self.collection.count()
            
            # Get all records metadata to analyze
            metadata_results = self.collection.get(include=["metadatas"])
            
            artists = set()
            albums = set()
            sources = set()
            
            if metadata_results['metadatas']:
                for metadata in metadata_results['metadatas']:
                    artists.add(metadata.get('artist', ''))
                    albums.add(metadata.get('album', ''))
                    sources.add(metadata.get('lyrics_source', ''))
            
            return {
                'total_songs': count,
                'unique_artists': len(artists),
                'unique_albums': len(albums),
                'lyrics_sources': list(sources),
                'embedding_model': self.model_name
            }
            
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {} 