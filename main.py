"""
main.py — Entry point for Muud desktop application.
Run from the project root:  python main.py
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"      # Force CPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"        # Suppress TF info logs

from engine.model_registry import ModelRegistry
from engine.recommender import MusicRecommender
from ui.desktop_app import MuudApp


def main():
    project_root = os.path.dirname(os.path.abspath(__file__))

    # Load all models exactly once via the singleton registry
    print("Loading models …")
    registry = ModelRegistry(project_root)
    registry.warmup()                       # JIT-compile TF graphs
    print("Models ready.")

    # Recommender receives pre-loaded models through the registry
    recommender = MusicRecommender(project_root)
    print("Launching UI …")

    app = MuudApp(recommender)
    app.run()


if __name__ == "__main__":
    main()
