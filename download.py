import os
import urllib.request

BASE = "https://cdn.jsdelivr.net/npm/d3-celestial@0.7.35/data/"
FILES = [
    "stars.6.json",
    "constellations.json",
    "constellations.lines.json",
]

def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "static", "data")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    for fn in FILES:
        url = BASE + fn
        dst = os.path.join(out_dir, fn)
        print(f"Downloading {url} -> {dst}")
        urllib.request.urlretrieve(url, dst)

    print("\nDone.")
    print("You should now have:")
    for fn in FILES:
        print(" -", os.path.join(out_dir, fn))

if __name__ == "__main__":
    main()
