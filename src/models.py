from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class SearchType(str, Enum):
    """Types of search available"""
    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


class Song(BaseModel):
    """Song model with metadata"""
    song_id: str
    original_id: str
    title: str
    artist: str
    album: Optional[str] = None
    isrc_id: Optional[str] = None
    duration_ms: Optional[int] = None
    popularity: Optional[int] = None
    release_date: Optional[str] = None
    added_at: datetime = Field(default_factory=datetime.now)
    source: str # "spotify" [to be extended to other API sources in the future]


class Lyrics(BaseModel):
    """Lyrics model"""
    song_id: str
    lyrics_text: str
    language: Optional[str] = None
    source: str  # "lclib" [possible future alternatives: "genius", "musixmatch", "manual"]
    last_updated: datetime = Field(default_factory=datetime.now)


class SongWithLyrics(BaseModel):
    """Combined song and lyrics data"""
    song: Song
    lyrics: Optional[Lyrics] = None
    embedding: Optional[List[float]] = None


class SearchQuery(BaseModel):
    """Search query model"""
    query: str
    search_type: SearchType = SearchType.HYBRID
    max_results: int = 20
    filters: Optional[Dict[str, Any]] = None


class SearchResult(BaseModel):
    """Search result model"""
    song: Song
    lyrics_excerpt: str
    relevance_score: float
    match_type: str  # "keyword", "semantic", "hybrid"
    matched_terms: List[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    """Response model for search results"""
    results: List[SearchResult]
    total_found: int
    query: str
    search_type: SearchType
    processing_time: float


class PlaylistImport(BaseModel):
    """Model for playlist import requests"""
    playlist_id: str
    playlist_name: Optional[str] = None
    fetch_lyrics: bool = True
    generate_embeddings: bool = True 