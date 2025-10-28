# Getting Started with LyriXplorer

## Prerequisites

- Python 3.8 or higher
- Spotify account (for playlist access)


## Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd lyriXplorer
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

1. **Copy the environment template:**
   ```bash
   cp config/env_example.txt .env
   ```

2. **Set up Spotify API credentials:**
   - Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
   - Create a new app
   - Copy the Client ID and Client Secret
   - Add `http://localhost:3000/callback` to Redirect URIs

3. **Edit your `.env` file:**
   ```env
   SPOTIFY_CLIENT_ID=your_spotify_client_id
   SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
   ```

## Running the Application

1. **Start the API server:**
   ```bash
   python src/main.py
   ```
   The API will be available at `http://localhost:8000`

2. **Open the web interface:**
   - Open `src/ui/index.html` in your browser
   - Or serve it with a simple HTTP server:
     ```bash
     cd src/ui
     python -m http.server 3000
     ```
   - Then visit `http://localhost:3000`

## First Steps

1. **Import your Spotify playlists:**
   - On startup, the app will retrieve your playlist content (songs and lyrics, if available)
   - Warning: It may take time to retrieve all your songs...
   - At each startup, the app will try to update your local DB with new songs and lyrics, if your playlists have changed

2. **Search for songs:**
   - Try searching for specific lyrics: "I want to hold your hand"
   - Try searching for emotions: "feeling sad and lonely"
   - Try searching for concepts: "love and heartbreak"

## API Endpoints

- `GET /` - Welcome message
- `GET /health` - Health check and stats
- `GET /playlists` - Get user's Spotify playlists
- `POST /search` - Search for songs
- `GET /stats` - Get database statistics

## Example Usage

### Search for songs:
```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "feeling happy and free",
    "search_type": "hybrid",
    "max_results": 10
  }'
```

## Troubleshooting

### Common Issues

1. **Spotify authentication fails:**
   - Check your Client ID and Secret
   - Ensure redirect URI is correct
   - Try clearing browser cookies

2. **No lyrics found:**
   - Some songs may not have lyrics available
   - Try different search terms

3. **Slow performance:**
   - First run will be slower due to model downloads and data fetching
   - Large playlists take time to process
   - Consider reducing max results

4. **Memory issues:**
   - Reduce batch size for large playlists
   - Close other applications
   - Consider using a smaller embedding model

### Getting Help

- Check the logs in the terminal for error messages
- Verify all API keys are correctly set
- Ensure you have sufficient disk space for embeddings

## Next Steps

- Explore the codebase to understand the architecture
- Customize the search algorithms
- Add your own lyrics sources
- Implement advanced features like playlist recommendations 