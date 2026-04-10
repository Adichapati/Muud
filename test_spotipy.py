import os
from dotenv import load_dotenv

_env_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".env"
)
load_dotenv(os.path.normpath(_env_path))


def _main() -> None:
    try:
        import spotipy
        from spotipy.oauth2 import SpotifyClientCredentials
    except ImportError:
        print("spotipy not installed. Install with: pip install spotipy")
        return

    try:
        auth_manager = SpotifyClientCredentials(
            client_id=os.environ.get("SPOTIPY_CLIENT_ID"),
            client_secret=os.environ.get("SPOTIPY_CLIENT_SECRET"),
        )
        sp = spotipy.Spotify(auth_manager=auth_manager)
        print("Auth ok.")

        print("Testing rock...")
        res = sp.search(q="genre:rock", type="track", limit=50)
        print("Rock works:", len(res["tracks"]["items"]))

        print("Testing Experimental...")
        res = sp.search(q="genre:Experimental", type="track", limit=50)
        print("Experimental works:", len(res["tracks"]["items"]))
    except Exception as e:
        print("Exception:", e)


if __name__ == "__main__":
    _main()
