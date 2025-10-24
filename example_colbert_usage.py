#!/usr/bin/env python3
"""
Example script demonstrating ColBERT embedding functionality in LyriXplorer.

This script shows how to:
1. Use ColBERT for both song-level and verse-level search
2. Switch between embedding approaches
3. Compare results between different methods
"""

import os
import sys
import logging

# Add src to path
sys.path.append('src')

from models import Song, Lyrics
from embeddings.embedding_factory import create_embedding_manager
from embeddings.colbert_manager import ColBERTManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_colbert_basic():
    """Basic ColBERT usage example"""
    print("\n=== ColBERT Basic Usage ===")
    
    # Create ColBERT manager
    colbert_manager = ColBERTManager(persist_directory="./colbert_embeddings")
    
    # Sample song data
    song = Song(
        song_id="demo_song_1",
        original_id="spotify_demo_1",
        title="Love and Moonlight",
        artist="Demo Artist",
        album="Demo Album",
        source="spotify"
    )
    
    lyrics = Lyrics(
        song_id="demo_song_1",
        lyrics_text="""
        I love the way you look tonight
        Under the moonlight so bright
        Your eyes are shining like the stars
        You're my everything, you're my heart
        
        And when the morning comes around
        I'll still be here, safe and sound
        With you by my side, everything's right
        You're my sunshine, you're my light
        """.strip(),
        source="manual"
    )
    
    # Add song with lyrics
    print("Adding song with lyrics...")
    success = colbert_manager.add_song_with_lyrics(song, lyrics)
    print(f"Song added: {success}")
    
    # Add verse embeddings
    print("Adding verse embeddings...")
    verse_count = colbert_manager.add_verse_embeddings(song, lyrics, include_overlapping=True)
    print(f"Added {verse_count} verse embeddings")
    
    # Search examples
    queries = [
        "love and moonlight",
        "shining stars",
        "morning sunshine",
        "emotional connection"
    ]
    
    for query in queries:
        print(f"\nSearching for: '{query}'")
        
        # Song-level search
        song_results = colbert_manager.search_songs(query, n_results=3)
        print(f"Song results ({len(song_results)}):")
        for result in song_results:
            print(f"  - {result['title']} by {result['artist']} (score: {result['similarity_score']:.3f})")
        
        # Verse-level search
        verse_results = colbert_manager.search_verses(query, n_results=3)
        print(f"Verse results ({len(verse_results)}):")
        for result in verse_results:
            print(f"  - Verse {result.get('verse_index', '?')}: {result['lyrics_excerpt'][:50]}... (score: {result['similarity_score']:.3f})")
        
        # Hybrid search
        hybrid_results = colbert_manager.hybrid_search(query, n_results=3)
        print(f"Hybrid results ({len(hybrid_results)}):")
        for result in hybrid_results:
            match_type = result.get('doc_type', 'unknown')
            print(f"  - {match_type}: {result['title']} - {result['lyrics_excerpt'][:50]}... (score: {result['hybrid_score']:.3f})")


def example_factory_usage():
    """Example of using the embedding factory"""
    print("\n=== Embedding Factory Usage ===")
    
    # Set environment variable to choose embedding type
    os.environ["EMBEDDING_TYPE"] = "colbert"
    
    # Create embedding manager through factory
    embedding_manager = create_embedding_manager("./factory_embeddings")
    print(f"Created embedding manager: {type(embedding_manager).__name__}")
    
    # Test basic functionality
    song = Song(
        song_id="factory_test",
        original_id="test_1",
        title="Test Song",
        artist="Test Artist",
        source="test"
    )
    
    lyrics = Lyrics(
        song_id="factory_test",
        lyrics_text="This is a test song with beautiful lyrics about love and dreams.",
        source="test"
    )
    
    # Add song
    success = embedding_manager.add_song_with_lyrics(song, lyrics)
    print(f"Song added via factory: {success}")
    
    # Search
    results = embedding_manager.search_similar("beautiful love", n_results=1)
    print(f"Search results: {len(results)} found")
    for result in results:
        print(f"  - {result['title']} (score: {result['similarity_score']:.3f})")


def example_comparison():
    """Compare different embedding approaches"""
    print("\n=== Embedding Approach Comparison ===")
    
    # Test query
    query = "love and moonlight"
    
    # Sample data
    song = Song(
        song_id="comparison_test",
        original_id="comp_1",
        title="Moonlight Love",
        artist="Test Artist",
        source="test"
    )
    
    lyrics = Lyrics(
        song_id="comparison_test",
        lyrics_text="""
        In the moonlight, I see your face
        Love is written in every trace
        Your heart beats with mine tonight
        Everything feels so right
        """.strip(),
        source="test"
    )
    
    # Test ColBERT
    print("Testing ColBERT...")
    os.environ["EMBEDDING_TYPE"] = "colbert"
    colbert_manager = create_embedding_manager("./comparison_colbert")
    colbert_manager.add_song_with_lyrics(song, lyrics)
    colbert_results = colbert_manager.search_similar(query, n_results=1)
    
    print(f"ColBERT results: {len(colbert_results)}")
    for result in colbert_results:
        print(f"  Score: {result['similarity_score']:.3f}")
    
    # Test SentenceTransformer
    print("\nTesting SentenceTransformer...")
    os.environ["EMBEDDING_TYPE"] = "sentence_transformer"
    st_manager = create_embedding_manager("./comparison_st")
    st_manager.add_song_with_lyrics(song, lyrics)
    st_results = st_manager.search_similar(query, n_results=1)
    
    print(f"SentenceTransformer results: {len(st_results)}")
    for result in st_results:
        print(f"  Score: {result['similarity_score']:.3f}")


def main():
    """Run all examples"""
    print("LyriXplorer ColBERT Embedding Examples")
    print("=" * 50)
    
    try:
        # Example 1: Basic ColBERT usage
        example_colbert_basic()
        
        # Example 2: Factory usage
        example_factory_usage()
        
        # Example 3: Comparison
        example_comparison()
        
        print("\n=== All examples completed successfully! ===")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
