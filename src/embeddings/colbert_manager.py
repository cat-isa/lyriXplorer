import os
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings
import logging
from models import Song, Lyrics, SongWithLyrics
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)


class ColBERTManager:
    """ColBERT-based embedding manager for both song and verse-level search"""
    
    def __init__(self, persist_directory: str = "./stored_embeddings", 
                 model_name: str = "colbert-ir/colbertv2.0"):
        self.persist_directory = persist_directory
        self.model_name = model_name
        
        # Initialize ColBERT model
        logger.info(f"Loading ColBERT model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Single collection for both songs and verses
        self.collection = self.client.get_or_create_collection(
            name="colbert_embeddings",
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"ColBERT manager initialized with {self.collection.count()} existing embeddings")
    
    def _encode_text(self, text: str, max_length: int = 512) -> Tuple[np.ndarray, np.ndarray]:
        """Encode text using ColBERT model"""
        try:
            # Tokenize
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=max_length,
                padding=True
            )
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Get token embeddings (batch_size, seq_len, hidden_size)
                token_embeddings = outputs.last_hidden_state.squeeze(0)  # Remove batch dimension
                
                # Get attention mask for valid tokens
                attention_mask = inputs['attention_mask'].squeeze(0)
                
                # Convert to numpy
                token_embeddings = token_embeddings.cpu().numpy()
                attention_mask = attention_mask.cpu().numpy()
                
                return token_embeddings, attention_mask
                
        except Exception as e:
            logger.error(f"Error encoding text: {e}")
            raise
    
    def _create_document_embedding(self, text: str, doc_type: str = "song") -> Dict[str, Any]:
        """Create ColBERT document embedding with metadata"""
        token_embeddings, attention_mask = self._encode_text(text)
        
        return {
            "token_embeddings": token_embeddings,
            "attention_mask": attention_mask,
            "text": text,
            "doc_type": doc_type,
            "num_tokens": int(attention_mask.sum())
        }
    
    def add_song_with_lyrics(self, song: Song, lyrics: Lyrics) -> bool:
        """Add a song with its lyrics to the vector database"""
        try:
            # Create song-level embedding
            song_embedding = self._create_document_embedding(lyrics.lyrics_text, "song")
            
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
                "lyrics_length": str(len(lyrics.lyrics_text)),
                "doc_type": "song"
            }
            
            # Store as JSON-serializable format
            embedding_data = {
                "token_embeddings": song_embedding["token_embeddings"].tolist(),
                "attention_mask": song_embedding["attention_mask"].tolist(),
                "num_tokens": song_embedding["num_tokens"]
            }
            
            # Add to ChromaDB
            self.collection.add(
                embeddings=[song_embedding["token_embeddings"].flatten()],  # Flatten for ChromaDB
                documents=[lyrics.lyrics_text],
                metadatas=[metadata],
                ids=[song.song_id]
            )
            
            logger.info(f"Added ColBERT embedding for song: {song.title} - {song.artist}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding song to vector database: {e}")
            return False
    
    def add_verse_embeddings(self, song: Song, lyrics: Lyrics, include_overlapping: bool = True) -> int:
        """Add verse-level embeddings for a song"""
        try:
            # Parse lyrics into verses
            verses = self._parse_lyrics_to_verses(lyrics.lyrics_text)
            verse_count = 0
            
            for i, verse_text in enumerate(verses):
                if self._is_valid_verse(verse_text):
                    # Create verse embedding
                    verse_embedding = self._create_document_embedding(verse_text, "verse")
                    
                    verse_id = f"{song.song_id}_verse_{i}"
                    metadata = {
                        "song_id": song.song_id,
                        "title": song.title,
                        "artist": song.artist,
                        "album": song.album or "",
                        "verse_id": verse_id,
                        "verse_index": str(i),
                        "verse_text": verse_text,
                        "doc_type": "verse"
                    }
                    
                    # Store verse
                    self.collection.add(
                        embeddings=[verse_embedding["token_embeddings"].flatten()],
                        documents=[verse_text],
                        metadatas=[metadata],
                        ids=[verse_id]
                    )
                    verse_count += 1
            
            # Add overlapping verse pairs if requested
            if include_overlapping and len(verses) > 1:
                for i in range(len(verses) - 1):
                    verse1, verse2 = verses[i], verses[i + 1]
                    if self._is_valid_verse(verse1) and self._is_valid_verse(verse2):
                        combined_text = f"{verse1}\n{verse2}"
                        overlap_embedding = self._create_document_embedding(combined_text, "verse_pair")
                        
                        overlap_id = f"{song.song_id}_overlap_{i}_{i+1}"
                        metadata = {
                            "song_id": song.song_id,
                            "title": song.title,
                            "artist": song.artist,
                            "album": song.album or "",
                            "verse_id": overlap_id,
                            "verse_index": str(i),
                            "verse_text": combined_text,
                            "doc_type": "verse_pair",
                            "parent_verses": f"{i},{i+1}"
                        }
                        
                        self.collection.add(
                            embeddings=[overlap_embedding["token_embeddings"].flatten()],
                            documents=[combined_text],
                            metadatas=[metadata],
                            ids=[overlap_id]
                        )
                        verse_count += 1
            
            logger.info(f"Added {verse_count} verse embeddings for song: {song.title}")
            return verse_count
            
        except Exception as e:
            logger.error(f"Error adding verse embeddings: {e}")
            return 0
    
    def _parse_lyrics_to_verses(self, lyrics_text: str) -> List[str]:
        """Parse lyrics into individual verses"""
        import re
        
        # Split by double newlines (common verse separator)
        verses = re.split(r'\n\s*\n', lyrics_text)
        
        # If no double newlines, try single newlines
        if len(verses) == 1:
            verses = lyrics_text.split('\n')
        
        # Clean up verses
        cleaned_verses = []
        for verse in verses:
            cleaned_verse = re.sub(r'\s+', ' ', verse.strip())
            if cleaned_verse:
                cleaned_verses.append(cleaned_verse)
        
        return cleaned_verses
    
    def _is_valid_verse(self, verse_text: str, min_length: int = 20, max_length: int = 500) -> bool:
        """Check if a verse is valid for embedding"""
        if not verse_text or not verse_text.strip():
            return False
        
        length = len(verse_text.strip())
        return min_length <= length <= max_length
    
    def search_similar(self, query: str, n_results: int = 10, 
                      filters: Optional[Dict[str, Any]] = None,
                      doc_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for similar content using ColBERT"""
        try:
            # Encode query
            query_embeddings, query_mask = self._encode_text(query)
            
            # Prepare where clause for filters
            where_clause = None
            if filters or doc_type:
                where_clause = {}
                if filters:
                    where_clause.update(filters)
                if doc_type:
                    where_clause["doc_type"] = doc_type
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embeddings.flatten()],
                n_results=n_results,
                where=where_clause,
                include=["metadatas", "documents", "distances"]
            )
            
            # Format results
            formatted_results = []
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    metadata = results['metadatas'][0][i]
                    formatted_results.append({
                        'song_id': metadata.get('song_id', ''),
                        'verse_id': metadata.get('verse_id', ''),
                        'title': metadata.get('title', ''),
                        'artist': metadata.get('artist', ''),
                        'album': metadata.get('album', ''),
                        'lyrics_excerpt': results['documents'][0][i],
                        'similarity_score': 1 - results['distances'][0][i],
                        'doc_type': metadata.get('doc_type', ''),
                        'verse_index': metadata.get('verse_index', ''),
                        'metadata': metadata
                    })
            
            logger.info(f"Found {len(formatted_results)} similar items for query: {query}")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching similar content: {e}")
            return []
    
    def search_verses(self, query: str, n_results: int = 10, 
                     filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search specifically for verses"""
        return self.search_similar(query, n_results, filters, doc_type="verse")
    
    def search_songs(self, query: str, n_results: int = 10, 
                     filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search specifically for songs"""
        return self.search_similar(query, n_results, filters, doc_type="song")
    
    def hybrid_search(self, query: str, n_results: int = 10, 
                     verse_weight: float = 0.7,
                     filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Hybrid search combining verse and song results"""
        try:
            # Get verse results
            verse_results = self.search_verses(query, n_results * 2, filters)
            
            # Get song results  
            song_results = self.search_songs(query, n_results, filters)
            
            # Combine and score
            all_results = []
            
            # Add verse results with higher weight
            for result in verse_results:
                result['hybrid_score'] = result['similarity_score'] * verse_weight
                result['match_type'] = 'verse'
                all_results.append(result)
            
            # Add song results with lower weight
            for result in song_results:
                result['hybrid_score'] = result['similarity_score'] * (1 - verse_weight)
                result['match_type'] = 'song'
                all_results.append(result)
            
            # Sort by hybrid score and remove duplicates
            seen_songs = set()
            unique_results = []
            
            for result in sorted(all_results, key=lambda x: x['hybrid_score'], reverse=True):
                song_id = result['song_id']
                if song_id not in seen_songs or result.get('verse_id'):
                    unique_results.append(result)
                    seen_songs.add(song_id)
            
            return unique_results[:n_results]
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return []
    
    def delete_song(self, song_id: str) -> bool:
        """Delete a song and all its verses from the database"""
        try:
            # Delete song
            self.collection.delete(ids=[song_id])
            
            # Delete all verses for this song
            verse_results = self.collection.get(
                where={"song_id": song_id, "doc_type": {"$in": ["verse", "verse_pair"]}},
                include=["ids"]
            )
            
            if verse_results['ids']:
                self.collection.delete(ids=verse_results['ids'])
            
            logger.info(f"Deleted song and verses from database: {song_id}")
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
            doc_types = {"song": 0, "verse": 0, "verse_pair": 0}
            
            if metadata_results['metadatas']:
                for metadata in metadata_results['metadatas']:
                    artists.add(metadata.get('artist', ''))
                    albums.add(metadata.get('album', ''))
                    sources.add(metadata.get('lyrics_source', ''))
                    doc_type = metadata.get('doc_type', 'song')
                    if doc_type in doc_types:
                        doc_types[doc_type] += 1
            
            return {
                'total_embeddings': count,
                'songs': doc_types['song'],
                'verses': doc_types['verse'],
                'verse_pairs': doc_types['verse_pair'],
                'unique_artists': len(artists),
                'unique_albums': len(albums),
                'lyrics_sources': list(sources),
                'embedding_model': self.model_name
            }
            
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}
