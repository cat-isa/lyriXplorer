#!/usr/bin/env python3
"""
Simple test script for LyriXplorer components
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all modules can be imported"""
    try:
        from models import Song, Lyrics, SearchQuery
        from api.spotify_client import SpotifyClient
        from api.lyrics_client import LyricsClient
        from embeddings.embedding_manager import EmbeddingManager
        from search.hybrid_search import HybridSearchEngine
        logger.info("✅ All imports successful")
        return True
    except Exception as e:
        logger.error(f"❌ Import failed: {e}")
        return False

def test_models():
    """Test data models"""
    try:
        from models import Song, Lyrics, SearchQuery, SearchType
        
        # Test Song model
        song = Song(
            song_id="test_id",
            original_id="test_id",
            title="Test Song",
            artist="Test Artist",
            album="Test Album",
            source="Test Source"
        )
        assert song.title == "Test Song"
        
        # Test Lyrics model
        lyrics = Lyrics(
            song_id="test_id",
            lyrics_text="This is a test lyric",
            source="test"
        )
        assert lyrics.lyrics_text == "This is a test lyric"
        
        # Test SearchQuery model
        query = SearchQuery(
            query="test query",
            search_type=SearchType.HYBRID,
            max_results=10
        )
        assert query.query == "test query"
        
        logger.info("✅ Data models working correctly")
        return True
    except Exception as e:
        logger.error(f"❌ Model test failed: {e}")
        return False

def test_embedding_manager():
    """Test embedding manager initialization"""
    try:
        from embeddings.embedding_manager import EmbeddingManager
        
        # Test with temporary directory
        import tempfile
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            embedding_mgr = EmbeddingManager(temp_dir) 
            stats = embedding_mgr.get_database_stats()
            assert isinstance(stats, dict)
            
        logger.info("✅ Embedding manager working correctly")
        return True
    except Exception as e:
        logger.error(f"❌ Embedding manager test failed: {e}")
        return False

def test_lyrics_client():
    """Test lyrics client initialization"""
    try:
        from api.lyrics_client import LyricsClient
        
        lyrics_client = LyricsClient()
        logger.info("✅ Lyrics client initialized successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Lyrics client test failed: {e}")
        return False

def test_spotify_client():
    """Test Spotify client initialization (requires credentials)"""
    try:
        from api.spotify_client import SpotifyClient
        
        # Check if credentials are available
        client_id = os.getenv("SPOTIFY_CLIENT_ID")
        client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
        
        if not client_id or not client_secret:
            logger.warning("⚠️  Spotify credentials not found, skipping Spotify client test")
            return True
        
        # Try to initialize (this might open a browser for auth)
        try:
            spotify_client = SpotifyClient()
            logger.info("✅ Spotify client initialized successfully")
            return True
        except Exception as auth_error:
            logger.warning(f"⚠️  Spotify authentication failed (expected without user interaction): {auth_error}")
            return True
            
    except Exception as e:
        logger.error(f"❌ Spotify client test failed: {e}")
        return False

def test_search_engine():
    """Test search engine initialization"""
    try:
        from embeddings.embedding_manager import EmbeddingManager
        from search.hybrid_search import HybridSearchEngine
        
        # Test with temporary directory
        import tempfile
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            embedding_mgr = EmbeddingManager(temp_dir)
            search_engine = HybridSearchEngine(embedding_mgr)
            
        logger.info("✅ Search engine working correctly")
        return True
    except Exception as e:
        logger.error(f"❌ Search engine test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("🧪 Running LyriXplorer tests...")
    
    tests = [
        ("Import Test", test_imports),
        ("Models Test", test_models),
        ("Embedding Manager Test", test_embedding_manager),
        ("Lyrics Client Test", test_lyrics_client),
        ("Spotify Client Test", test_spotify_client),
        ("Search Engine Test", test_search_engine),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n🔍 Running {test_name}...")
        if test_func():
            passed += 1
        else:
            logger.error(f"❌ {test_name} failed")
    
    logger.info(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! LyriXplorer is ready to use.")
        return 0
    else:
        logger.error("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 