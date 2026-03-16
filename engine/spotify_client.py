"""
spotify_client.py
-----------------
Spotify API integration for MUUD.

Fetches real track data from Spotify so the recommender can supplement
or replace the static song_db.csv with live search results.

Requires:
    pip install spotipy

Environment variables (or pass directly):
    SPOTIPY_CLIENT_ID
    SPOTIPY_CLIENT_SECRET
"""

import os
import logging
from typing import Optional

from dotenv import load_dotenv

# Auto-load .env from project root (Muud/.env)
_env_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, ".env"
)
load_dotenv(os.path.normpath(_env_path))

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

logger = logging.getLogger(__name__)

# Silence spotipy's noisy HTTP error logging — we handle errors gracefully
logging.getLogger("spotipy").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)

class SpotifyClient:
    """Lightweight wrapper around the Spotify Web API via Spotipy."""

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
    ):
        """
        Authenticate with Spotify using Client Credentials flow.

        Args:
            client_id:     Spotify app client ID.
                           Falls back to env var SPOTIPY_CLIENT_ID.
            client_secret: Spotify app client secret.
                           Falls back to env var SPOTIPY_CLIENT_SECRET.
        """
        cid = client_id or os.environ.get("SPOTIPY_CLIENT_ID")
        csecret = client_secret or os.environ.get("SPOTIPY_CLIENT_SECRET")

        if not cid or not csecret:
            raise ValueError(
                "Spotify credentials not found. "
                "Set SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET "
                "environment variables, or pass them directly."
            )

        auth_manager = SpotifyClientCredentials(
            client_id=cid,
            client_secret=csecret,
        )
        self.sp = spotipy.Spotify(auth_manager=auth_manager)
        logger.info("SpotifyClient authenticated successfully.")

    # ── Public API ───────────────────────────────────────────────

    def search_tracks_by_genre(
        self, genre: str, limit: int = 50
    ) -> list[dict]:
        """Search Spotify for tracks matching a genre tag.

        Args:
            genre: Genre string (e.g. "rock", "classical", "hip-hop").
            limit: Max number of tracks to return (1-50).

        Returns:
            List of normalised track dicts.
        """
        clean_genre = genre.replace(" ", "").lower()
        strict_query = f"genre:{clean_genre}"
        
        # 1. Try strict genre search
        tracks = self._search(strict_query, limit)
        
        # 2. If nothing found or API rejected it, fallback to keyword search
        if not tracks:
            logger.info("Spotify API strictly rejected genre search. Fallback to keyword: '%s'", genre)
            tracks = self._search(genre, limit)
            
        return tracks

    def search_tracks_by_query(
        self, query: str, limit: int = 50
    ) -> list[dict]:
        """Generic Spotify search (artist, track name, keywords, etc.).

        Args:
            query: Free-text search query.
            limit: Max number of tracks to return (1-50).

        Returns:
            List of normalised track dicts.
        """
        return self._search(query, limit)

    # ── Internals ────────────────────────────────────────────────

    def _search(self, query: str, limit: int) -> list[dict]:
        """Execute a Spotify search and normalise the results."""
        # Spotify API has a bug where limit > 20 for 'genre:' searches
        # often throws a 400 Invalid Limit without a market parameter.
        limit = max(1, min(limit, 20))

        try:
            # Drop the limit and market variables entirely to see if spotipy serializes the base URL successfully
            response = self.sp.search(q=query, type="track")
        except Exception as exc:
            # We catch Exception instead of SpotifyException to ensure the fallback ALWAYS triggers
            logger.error("Spotify search failed internally. Triggering fallback. %s", exc)
            return []

        items = response.get("tracks", {}).get("items", [])
        return [self._normalise_track(item) for item in items]

    @staticmethod
    def _normalise_track(item: dict) -> dict:
        """Convert a raw Spotify track object into the MUUD format.

        Output schema (matches what the recommender expects):
            song         – track name
            artist       – primary artist name
            preview_url  – 30-second MP3 preview (may be None)
            album_art    – largest album cover image URL
            spotify_url  – external Spotify link
            popularity   – Spotify popularity score (0-100)
        """
        artists = item.get("artists", [])
        artist_name = artists[0]["name"] if artists else "Unknown"

        album = item.get("album", {})
        images = album.get("images", [])
        # images are sorted largest → smallest; take the first
        album_art = images[0]["url"] if images else None

        external_urls = item.get("external_urls", {})

        return {
            "song": item.get("name", "Unknown"),
            "artist": artist_name,
            "preview_url": item.get("preview_url"),
            "album_art": album_art,
            "spotify_url": external_urls.get("spotify"),
            "popularity": item.get("popularity", 0),
        }
