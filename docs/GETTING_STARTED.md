# Getting Started with LyriXplorer

## Prerequisites

- Python 3.8 or higher
- Spotify account (for playlist access)
- Optional: Genius API token (for better lyrics fetching)
- Optional: Musixmatch API key (alternative lyrics source)

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

3. **Optional: Set up Genius API (recommended):**
   - Go to [Genius API](https://genius.com/api-clients)
   - Create an API client
   - Copy the access token

4. **Optional: Set up Musixmatch API:**
   - Go to [Musixmatch Developer](https://developer.musixmatch.com/)
   - Sign up and get an API key

5. **Edit your `.env` file:**
   ```env
   SPOTIFY_CLIENT_ID=your_spotify_client_id
   SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
   GENIUS_ACCESS_TOKEN=your_genius_token
   MUSIXMATCH_API_KEY=your_musixmatch_key
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

1. **Import a Spotify playlist:**
   - Get a playlist ID from Spotify (the part after `/playlist/` in the URL)
   - Use the import form in the web interface
   - The app will fetch all tracks and lyrics

2. **Search for songs:**
   - Try searching for specific lyrics: "I want to hold your hand"
   - Try searching for emotions: "feeling sad and lonely"
   - Try searching for concepts: "love and heartbreak"

## API Endpoints

- `GET /` - Welcome message
- `GET /health` - Health check and stats
- `GET /playlists` - Get user's Spotify playlists
- `POST /import-playlist` - Import a playlist
- `POST /search` - Search for songs
- `GET /stats` - Get database statistics
- `DELETE /songs/{song_id}` - Delete a song

## Example Usage

### Import a playlist via API:
```bash
curl -X POST "http://localhost:8000/import-playlist" \
  -H "Content-Type: application/json" \
  -d '{
    "playlist_id": "37i9dQZF1DXcBWIGoYBM5M",
    "fetch_lyrics": true,
    "generate_embeddings": true
  }'
```

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
   - Check if Genius/Musixmatch APIs are configured
   - Some songs may not have lyrics available
   - Try different search terms

3. **Slow performance:**
   - First run will be slower due to model downloads
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