import os
import logging
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import uvicorn

from models import SearchQuery, SearchResponse
from api.spotify_client import SpotifyClient
from api.lyrics_client import LyricsClient
from embeddings.embedding_factory import create_embedding_manager
from search.hybrid_search import HybridSearchEngine
from services.sync_service import SyncService
from services.lyrics_fetch_service import LyricsFetchService

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="LyriXplorer API",
    description="A personal lyrics-based song discovery app",
    version="1.0.0"
)

# Add CORS middleware (configurable)
cors_origins_env = os.getenv("CORS_ORIGINS", "*")
cors_origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()]
allow_credentials = os.getenv("CORS_ALLOW_CREDENTIALS", "false").lower() == "true"

if cors_origins == ["*"]:
    # When using wildcard with credentials, browsers reject; use regex instead
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=".*",
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Initialize components
spotify_client = None
lyrics_client = None
embedding_manager = None
search_engine = None
sync_service = None
lyrics_service = None

def get_spotify_client():
    global spotify_client
    if spotify_client is None:
        try:
            spotify_client = SpotifyClient()
        except Exception as e:
            logger.error(f"Failed to initialize Spotify client: {e}")
            raise HTTPException(status_code=500, detail="Spotify client initialization failed")
    return spotify_client

def get_lyrics_client():
    global lyrics_client
    if lyrics_client is None:
        lyrics_client = LyricsClient()
    return lyrics_client

def get_embedding_manager():
    global embedding_manager
    if embedding_manager is None:
        persist_dir = os.getenv("CHROMA_PERSIST_DIRECTORY", "./stored_embeddings")
        embedding_manager = create_embedding_manager(persist_dir)
    return embedding_manager

def get_search_engine():
    global search_engine
    if search_engine is None:
        persist_dir = os.getenv("KEYWORD_INDEX_PERSIST_DIRECTORY", "./keyword_index")
        data_dir = os.getenv("DATA_DIR", "./data")
        enable_lemmatization = os.getenv("ENABLE_LEMMATIZATION", True)
        enable_fuzzy_matching = os.getenv("ENABLE_FUZZY_MATCHING", True)
        enable_phrase_matching = os.getenv("ENABLE_PHRASE_MATCHING", True)
        embedding_mgr = get_embedding_manager()
        search_engine = HybridSearchEngine(embedding_mgr, enable_lemmatization, enable_fuzzy_matching, enable_phrase_matching, 
                                            data_dir, persist_dir)
    return search_engine

def get_sync_service():
    global sync_service
    if sync_service is None:
        data_dir = os.getenv("DATA_DIR", "./data")
        sync_service = SyncService(data_dir)
    return sync_service

def get_lyrics_service():
    global lyrics_service
    if lyrics_service is None:
        data_dir = os.getenv("DATA_DIR", "./data")
        lyrics_service = LyricsFetchService(data_dir)
    return lyrics_service

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    logger.info("Starting LyriXplorer API...")
    
    # Test component initialization
    try:
        get_spotify_client()
        get_lyrics_client()
        get_embedding_manager()
        get_search_engine()
        get_lyrics_service()
        logger.info("All components initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")

    # Optional startup sync
    try:
        if os.getenv("SYNC_ON_STARTUP", "false").lower() == "true":
            svc = get_sync_service()
            summary = svc.sync_all(get_spotify_client(), get_lyrics_client(), get_lyrics_service(), get_search_engine())
            logger.info(f"Startup sync completed: {summary}")
    except Exception as e:
        logger.error(f"Startup sync failed: {e}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to LyriXplorer API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test basic functionality
        embedding_mgr = get_embedding_manager()
        stats = embedding_mgr.get_database_stats()
        
        return {
            "status": "healthy",
            "database_stats": stats,
            "components": {
                "spotify_client": spotify_client is not None,
                "lyrics_client": lyrics_client is not None,
                "embedding_manager": embedding_manager is not None,
                "search_engine": search_engine is not None,
                "sync_service": sync_service is not None,
                "lyrics_service": lyrics_service is not None
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

@app.get("/playlists")
async def get_playlists(spotify: SpotifyClient = Depends(get_spotify_client)):
    """Get user's Spotify playlists"""
    try:
        playlists = spotify.get_user_playlists()
        return {"playlists": playlists}
    except Exception as e:
        logger.error(f"Error fetching playlists: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch playlists")

@app.post("/sync")
async def sync_library(
    svc: SyncService = Depends(get_sync_service),
    spotify: SpotifyClient = Depends(get_spotify_client),
    lyrics: LyricsClient = Depends(get_lyrics_client),
    lyrics_service: LyricsFetchService = Depends(get_lyrics_service),
    search_engine: HybridSearchEngine = Depends(get_search_engine)
):
    """Manually trigger a sync from Spotify to local CSV DB and fetch missing lyrics."""
    try:
        summary = svc.sync_all(spotify, lyrics, lyrics_service, search_engine) 
        return {"message": "Sync completed", "summary": summary}
    except Exception as e:
        logger.error(f"Sync failed: {e}")
        raise HTTPException(status_code=500, detail="Sync failed")

@app.post("/lyrics/start")
async def lyrics_start(
    lyrics_svc: LyricsFetchService = Depends(get_lyrics_service),
    lyrics_client: LyricsClient = Depends(get_lyrics_client),
    search_engine: HybridSearchEngine = Depends(get_search_engine)
):
    try:
        res = lyrics_svc.start(lyrics_client, search_engine)
        return res
    except Exception as e:
        logger.error(f"Lyrics start failed: {e}")
        raise HTTPException(status_code=500, detail="Lyrics start failed")

@app.post("/lyrics/pause")
async def lyrics_pause(lyrics_svc: LyricsFetchService = Depends(get_lyrics_service)):
    return lyrics_svc.pause()

@app.post("/lyrics/resume")
async def lyrics_resume(lyrics_svc: LyricsFetchService = Depends(get_lyrics_service)):
    return lyrics_svc.resume()

@app.post("/lyrics/cancel")
async def lyrics_cancel(lyrics_svc: LyricsFetchService = Depends(get_lyrics_service)):
    return lyrics_svc.cancel()

@app.get("/lyrics/status")
async def lyrics_status(lyrics_svc: LyricsFetchService = Depends(get_lyrics_service)):
    return lyrics_svc.status()

@app.post("/search", response_model=SearchResponse)
async def search_songs(
    search_query: SearchQuery,
    search_engine: HybridSearchEngine = Depends(get_search_engine)
):
    """Search for songs by lyrics content"""
    try:
        results = search_engine.search(search_query)
        return results
    except Exception as e:
        logger.error(f"Error in search: {e}")
        raise HTTPException(status_code=500, detail="Search failed")

@app.get("/stats")
async def get_stats(embedding_mgr: EmbeddingManager = Depends(get_embedding_manager)):
    """Get database statistics"""
    try:
        stats = embedding_mgr.get_database_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get statistics")

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("DEBUG", "False").lower() == "true"
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug,
        reload_dirs=["src"],
        log_level="info"
    ) 