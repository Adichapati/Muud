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
        query = f"genre:{genre}"
        return self._search(query, limit)

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
        limit = max(1, min(limit, 50))  # Spotify caps at 50 per request

        try:
            response = self.sp.search(q=query, type="track", limit=limit)
        except spotipy.SpotifyException as exc:
            logger.error("Spotify search failed: %s", exc)
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
