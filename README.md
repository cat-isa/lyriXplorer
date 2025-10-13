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
├── data/                   # CSV files with song metadata
├── lyrics/                 # Cached lyrics files
├── embeddings/             # Vector embeddings database
├── src/
│   ├── api/               # Spotify and lyrics APIs
│   ├── search/            # Search algorithms
│   ├── embeddings/        # Vector embedding generation
│   └── ui/                # Web interface
├── config/                # Configuration files
└── requirements.txt       # Python dependencies
```

## Tech Stack

- **Backend**: Python (FastAPI)
- **Frontend**: React/Next.js
- **Vector DB**: ChromaDB (local)
- **Embeddings**: sentence-transformers (multilingual)
- **APIs**: Spotify Web API, LRCLIB API
- **Data**: CSV, JSON

## Getting Started

1. Install dependencies: `pip install -r requirements.txt`
2. Set up API keys in `config/.env`
3. Import your Spotify playlists to CSV
4. Run the app: `python src/main.py`

## Development Phases

### Phase 1: Core Infrastructure
- [x] Project setup
- [ ] Spotify playlist import
- [ ] Lyrics fetching
- [ ] Basic search interface

### Phase 2: Smart Search
- [ ] Vector embeddings generation
- [ ] Semantic search implementation
- [ ] Hybrid search algorithm

### Phase 3: Enhanced Features
- [ ] LLM-powered search coordination
- [ ] Advanced filtering
- [ ] Search history and favorites

## License

MIT License - Personal use only 