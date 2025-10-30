# LyriXplorer 🎵

A personal app to look for songs in your playlists whose lyrics match emotional themes or specific terms

## Features

- **Lyrics Search**: Find songs containing specific text in lyrics
- **Semantic Search**: Discover songs expressing similar feelings or concepts
- **Personal Database**: Works with your own Spotify playlists
- **Multilingual Support**: Search across songs in different languages
- **Hybrid Search**: Combines keyword and semantic search for better results

## Architecture

```
lyriXplorer/
├── config/                 # Configuration files
├── data/                   # CSV files with song metadata
├── docs/                   # GETTING_STARTED.md
├── keyword_index/          # precomputed indexes for lexical search
├── src/
│   ├── api/                # Spotify and lyrics APIs
│   ├── embeddings/         # Vector embedding generation
│   ├── search/             # Search algorithms
│   ├── services/           # Lyrics fetching service and DB sync
│   ├── storage/            # DB implementation (CSV based)
│   ├── ui/                 # Web interface
│   ├── main.py                 # App
│   └── models.py               # Data models
├── stored_embeddings/      # Vector embeddings database
├── explore_data.ipynb      # Draft notebook to experiment with the code base
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── setup.py                # Setup script
└── test_app.py             # App setup tests
```

## Tech Stack

- **Backend**: Python (FastAPI)
- **Frontend**: HTML+JS
- **Vector DB**: ChromaDB (local)
- **Embeddings**: sentence-transformers (multilingual)
- **APIs**: Spotify Web API, LRCLIB API
- **Data**: CSV, JSON

## Getting Started

1. Install dependencies: `pip install -r requirements.txt`
2. Set up API keys in `.env`
3. Run the app: `python src/main.py`

## Development Phases

### Phase 1: Core Infrastructure
- [x] Project setup
- [x] Spotify playlist import
- [x] Lyrics fetching
- [x] Basic search interface

### Phase 2: Smart Search
- [x] Vector embeddings generation
- [x] Semantic search implementation
- [x] Hybrid search algorithm

### Phase 3: Enhanced Features & Improvements
- [ ] Verse-level embeddings generation
- [ ] Add a reranker?
- [ ] ColBERT embeddings generation
- [ ] LLM-powered query expansion
- [ ] Allow manual insertion of missing lyrics
- [ ] Analytics on song distribution, themes, ...
- [ ] Playlist generation from search results
- [ ] Add more tests, improve docstrings and logging
- [ ] Automatic language detection (to add as metadata for lyrics)
- [ ] At import time, let user select playlists to import/exclude
- [ ] Advanced filtering

## License

MIT License - Personal use only 