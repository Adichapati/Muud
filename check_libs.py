import importlib, sys

reqs = [
    'tensorflow', 'numpy', 'sklearn', 'librosa', 'sounddevice', 
    'scipy', 'pandas', 'matplotlib', 'seaborn', 'pygame', 
    'spotipy', 'customtkinter', 'dotenv', 'PIL'
]

print("\n--- DEPENDENCY CHECK ---")
missing = []
for req in reqs:
    try:
        importlib.import_module(req)
        print(f"[OK] {req}")
    except ImportError:
        missing.append(req)
        print(f"[MISSING] {req}")

print("\nSUMMARY:")
if missing:
    print(f"You need to install: pip install {' '.join(missing)}")
else:
    print("All required libraries are installed!")
