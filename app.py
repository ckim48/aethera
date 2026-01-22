# app.py
from __future__ import annotations

import os
import re
import json
import math
import unicodedata
from datetime import datetime, timezone
from urllib.parse import quote

import pandas as pd
from flask import Flask, render_template, jsonify, request, url_for

app = Flask(__name__)

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
DATA_DIR = os.path.join(app.root_path, "static", "data")
AUDIO_DIR = os.path.join(app.root_path, "static", "audio")

MAJOR_CSV = os.path.join(DATA_DIR, "star_map.csv")  # your audio subset CSV
CONSTELLATIONS_JSON = os.path.join(DATA_DIR, "constellations.json")  # GeoJSON abbr->name
CONST_LINES_JSON = os.path.join(DATA_DIR, "constellations.lines.json")  # GeoJSON polylines in RA/Dec

# Optional GPT summary (won't affect visibility filtering)
try:
    from openai import OpenAI
    _OPENAI_OK = True
except Exception:
    OpenAI = None
    _OPENAI_OK = False

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")

_openai_client = OpenAI(api_key=OPENAI_API_KEY) if (_OPENAI_OK and OPENAI_API_KEY) else None
def compute_visible_constellations(
    lat: float,
    lon: float,
    dt: datetime,
    min_alt_deg: float = 10.0,      # more realistic than -8 or 0
    max_star_mag: float = 4.5,      # hide very dim stars
    min_stars_per_const: int = 1,   # require at least N stars above threshold
) -> dict:
    """
    Uses your MAJOR_AUDIO list (audio subset) to decide which constellations are realistically visible.

    Returns:
      {
        "visible": [{"name": "...", "count": 3}, ...],
        "counts": {...}
      }
    """
    total_csv_const = len(CSV_CONST_NAMES_SET)

    counts_by_const: dict[str, int] = {}
    for s in MAJOR_AUDIO:
        const_name = (s.get("const_name") or "").strip()
        if not const_name:
            continue
        if _norm_key(const_name) not in CSV_CONST_KEYS:
            continue

        # magnitude filter (realistic visibility)
        try:
            mag = float(s.get("mag", 99.0))
        except Exception:
            mag = 99.0
        if mag > max_star_mag:
            continue

        ra_h = (float(s["ra_deg"]) / 15.0) % 24.0
        alt, az = alt_az_from_ra_dec(ra_h, float(s["dec_deg"]), lat, lon, dt)
        if alt < float(min_alt_deg):
            continue

        counts_by_const[const_name] = counts_by_const.get(const_name, 0) + 1

    visible = [
        {"name": k, "count": v}
        for k, v in counts_by_const.items()
        if v >= int(min_stars_per_const)
    ]
    visible.sort(key=lambda x: (-x["count"], x["name"].upper()))

    return {
        "visible": visible,
        "counts": {
            "csv_constellations_total": total_csv_const,
            "csv_audio_rows_total": len(MAJOR_AUDIO),
            "visible_constellations": len(visible),
        },
    }
@app.route("/api/visible_constellations")
def api_visible_constellations():
    lat = float(request.args.get("lat", 37.5665))
    lon = float(request.args.get("lon", 126.9780))
    ts = request.args.get("ts", "")
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00")) if ts else datetime.now(timezone.utc)

    # tweak these defaults anytime
    min_alt = float(request.args.get("min_alt", 10.0))
    max_mag = float(request.args.get("max_mag", 4.5))
    min_stars = int(request.args.get("min_stars", 1))

    result = compute_visible_constellations(
        lat=lat,
        lon=lon,
        dt=dt,
        min_alt_deg=min_alt,
        max_star_mag=max_mag,
        min_stars_per_const=min_stars,
    )

    visible = result["visible"]
    counts = result["counts"]

    # ---- CONSOLE PRINT (what you asked for) ----
    print(
        "[VISIBLE]",
        f"lat={lat:.4f} lon={lon:.4f} ts={dt.isoformat()} | "
        f"CSV constellations={counts['csv_constellations_total']} | "
        f"Audio rows={counts['csv_audio_rows_total']} | "
        f"Visible constellations={counts['visible_constellations']} "
        f"(min_alt={min_alt}, max_mag={max_mag}, min_stars={min_stars})"
    )

    # Optional GPT natural-language summary (does not change what is visible)
    gpt_text = None
    if _openai_client and visible:
        names = [v["name"] for v in visible[:30]]
        prompt = (
            "You are an astronomy guide. Given a location/time and a list of visible constellations, "
            "write a short, practical viewing summary (2-4 sentences) and then a comma-separated list "
            "of the top constellations to look for first.\n\n"
            f"Location: lat={lat}, lon={lon}\n"
            f"Time (UTC): {dt.isoformat()}\n"
            f"Visible constellations ({len(visible)}): {', '.join(names)}"
        )
        try:
            resp = _openai_client.responses.create(
                model=OPENAI_MODEL,
                input=prompt,
            )
            # responses API returns output items; simplest is to read .output_text if available
            gpt_text = getattr(resp, "output_text", None) or str(resp)
        except Exception as e:
            print("[VISIBLE][GPT] failed:", repr(e))
            gpt_text = None

    return jsonify(
        {
            "visible": visible,
            "counts": counts,
            "params": {"min_alt": min_alt, "max_mag": max_mag, "min_stars": min_stars},
            "ts_utc": dt.isoformat().replace("+00:00", "Z"),
            "gpt_summary": gpt_text,
        }
    )

# ------------------------------------------------------------
# Frontend-projection compatibility
# IMPORTANT: Must match sky.html for consistent geometry
# ------------------------------------------------------------
SKY_SCALE = 34.0  # same as sky.html

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def slugify(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-") or "star"


def _norm_key(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def load_json(path: str, label: str):
    if not os.path.exists(path):
        print(f"[WARN] Missing {label}: {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _as_feature_list(obj):
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and isinstance(obj.get("features"), list):
        return obj["features"]
    return []


# ------------------------------------------------------------
# Unicode + audio filename matching (robust)
# ------------------------------------------------------------
def ukey(s: str) -> str:
    """
    Normalize text for robust filename matching across OS/filesystems.
    - strip
    - remove trailing .mp3
    - Unicode normalize NFC
    - casefold
    - collapse whitespace
    """
    s = (s or "").strip()
    if s.lower().endswith(".mp3"):
        s = s[:-4]
    s = unicodedata.normalize("NFC", s)
    s = s.casefold()
    s = re.sub(r"\s+", " ", s)
    return s


def build_audio_index(audio_dir: str) -> dict[str, str]:
    """Map normalized key -> actual filename on disk (mp3 only)."""
    idx: dict[str, str] = {}
    if not os.path.isdir(audio_dir):
        return idx
    for fn in os.listdir(audio_dir):
        if not fn.lower().endswith(".mp3"):
            continue
        idx[ukey(fn)] = fn
    return idx


AUDIO_INDEX = build_audio_index(AUDIO_DIR)
print(f"[INFO] Audio files indexed: {len(AUDIO_INDEX)}")


def resolve_audio_filename(file_cell: str | None, star_name: str) -> str | None:
    """
    Resolve mp3 filename using:
      1) File column exact match
      2) Star Name exact match
      3) Fuzzy match for patterns like 'Mirfak (α Per).mp3' when CSV has 'Mirfak'
    """
    candidates: list[str] = []

    if file_cell and str(file_cell).strip():
        candidates.append(str(file_cell).strip())

    if star_name and star_name.strip():
        candidates.append(star_name.strip())

    # Exact matches first (with/without .mp3)
    for raw in candidates:
        hit = AUDIO_INDEX.get(ukey(raw))
        if hit:
            return hit
        if not raw.lower().endswith(".mp3"):
            hit = AUDIO_INDEX.get(ukey(raw + ".mp3"))
            if hit:
                return hit

    # Fuzzy: startswith / contains
    for raw in candidates:
        sk = ukey(raw)
        if not sk:
            continue

        starts = [fn for k, fn in AUDIO_INDEX.items() if k.startswith(sk)]
        if starts:
            return sorted(starts, key=lambda s: (len(s), s))[0]

        contains = [fn for k, fn in AUDIO_INDEX.items() if sk in k]
        if contains:
            return sorted(contains, key=lambda s: (len(s), s))[0]

    return None


# ------------------------------------------------------------
# CSV parsing (audio subset)
# ------------------------------------------------------------
def ra_hms_to_hours(ra) -> float:
    if ra is None or (isinstance(ra, float) and math.isnan(ra)):
        return 0.0
    parts = str(ra).strip().split(":")
    h = float(parts[0]) if len(parts) > 0 and parts[0] else 0.0
    m = float(parts[1]) if len(parts) > 1 and parts[1] else 0.0
    s = float(parts[2]) if len(parts) > 2 and parts[2] else 0.0
    return h + m / 60.0 + s / 3600.0


def dec_dms_to_deg(dec) -> float:
    if dec is None or (isinstance(dec, float) and math.isnan(dec)):
        return 0.0
    t = str(dec).strip()
    sign = -1 if t.startswith("-") else 1
    t = t.lstrip("+-")
    parts = t.split(":")
    d = float(parts[0]) if len(parts) > 0 and parts[0] else 0.0
    m = float(parts[1]) if len(parts) > 1 and parts[1] else 0.0
    s = float(parts[2]) if len(parts) > 2 and parts[2] else 0.0
    return sign * (d + m / 60.0 + s / 3600.0)


def load_major_csv(abs_path: str):
    if not os.path.exists(abs_path):
        print(f"[WARN] CSV not found: {abs_path}")
        return []

    df = pd.read_csv(abs_path)

    # strip headers (handles "File " etc.)
    df.columns = [c.strip() for c in df.columns]

    # fill constellation down
    if "Constellation" in df.columns:
        df["Constellation"] = df["Constellation"].ffill()

    df = df.where(pd.notnull(df), None)

    stars = []
    for _, r in df.iterrows():
        const = (r.get("Constellation") or "").strip() or "Unknown"
        name = (r.get("Star Name") or "").strip()
        if not name:
            continue

        ra_h = ra_hms_to_hours(r.get("RA (hh:mm:ss)") or r.get("RA") or "00:00:00")
        dec_d = dec_dms_to_deg(r.get("Dec (dd:mm:ss)") or r.get("Dec") or "+00:00:00")

        mag = r.get("Magnitude")
        try:
            mag = float(mag) if mag is not None else 6.0
        except Exception:
            mag = 6.0

        # robust audio resolution
        resolved_filename = resolve_audio_filename(r.get("File"), name)

        audio_url = None
        audio_exists = False
        if resolved_filename:
            abs_audio = os.path.join(AUDIO_DIR, resolved_filename)
            audio_exists = os.path.exists(abs_audio)
            audio_url = f"/static/audio/{quote(resolved_filename, safe='')}"

        stars.append(
            {
                "name": name,
                "const_name": const,
                "ra_deg": float(ra_h * 15.0),
                "dec_deg": float(dec_d),
                "mag": float(mag),
                "audio": audio_url,
                "audio_exists": bool(audio_exists),
                "audio_file": resolved_filename,  # debug
                "meta": r.to_dict(),
            }
        )

    return stars


MAJOR_AUDIO = load_major_csv(MAJOR_CSV)

CSV_CONST_NAMES_SET = set((s.get("const_name") or "").strip() for s in MAJOR_AUDIO)
CSV_CONST_NAMES_SET.discard("")
CSV_CONST_KEYS = set(_norm_key(n) for n in CSV_CONST_NAMES_SET)

print(f"[INFO] Loaded audio-subset CSV stars: {len(MAJOR_AUDIO)}")
print(f"[INFO] CSV constellations loaded: {len(CSV_CONST_NAMES_SET)}")
print(f"[INFO] Stars with audio URL: {sum(1 for s in MAJOR_AUDIO if s.get('audio'))}")
print(f"[INFO] Stars with mp3 exists: {sum(1 for s in MAJOR_AUDIO if s.get('audio_exists'))}")


# ------------------------------------------------------------
# constellations.json parsing (abbr->full name)
# ------------------------------------------------------------
def parse_constellations(const_raw):
    feats = _as_feature_list(const_raw)
    abbr_to_name = {}
    for f in feats:
        if not isinstance(f, dict):
            continue
        props = f.get("properties") or {}
        abbr = (f.get("id") or props.get("id") or props.get("desig") or "").strip()
        name = (props.get("name") or props.get("en") or "").strip()
        if abbr and name:
            abbr_to_name[abbr] = name
    return abbr_to_name


# ------------------------------------------------------------
# constellations.lines.json parsing (GeoJSON polylines in RA/Dec)
# ------------------------------------------------------------
def parse_constellation_lines_geojson(lines_raw):
    feats = _as_feature_list(lines_raw)
    out = {}
    for f in feats:
        if not isinstance(f, dict):
            continue
        props = f.get("properties") or {}
        abbr = (f.get("id") or props.get("id") or props.get("desig") or props.get("con") or "").strip()
        if not abbr:
            continue

        geom = f.get("geometry") or {}
        coords = geom.get("coordinates")
        gtype = (geom.get("type") or "").strip()

        if gtype == "LineString" and isinstance(coords, list):
            polylines = [coords]
        elif gtype == "MultiLineString" and isinstance(coords, list):
            polylines = coords
        else:
            polylines = coords if isinstance(coords, list) else []

        cleaned = []
        for line in polylines:
            if not (isinstance(line, list) and len(line) >= 2):
                continue
            pts = []
            ok = True
            for p in line:
                if not (isinstance(p, list) and len(p) >= 2):
                    ok = False
                    break
                try:
                    ra = float(p[0])
                    dec = float(p[1])
                except Exception:
                    ok = False
                    break
                pts.append([ra, dec])
            if ok and len(pts) >= 2:
                cleaned.append(pts)

        if cleaned:
            out[abbr] = cleaned

    return out


# ------------------------------------------------------------
# Load constellation datasets
# ------------------------------------------------------------
CONST_RAW = load_json(CONSTELLATIONS_JSON, "constellations.json")
LINES_RAW = load_json(CONST_LINES_JSON, "constellations.lines.json")

ABBR_TO_NAME = parse_constellations(CONST_RAW)
CONST_LINES = parse_constellation_lines_geojson(LINES_RAW)

print(f"[INFO] Constellation names loaded: {len(ABBR_TO_NAME)}")
print(f"[INFO] Constellation line sets loaded: {len(CONST_LINES)}")


# ------------------------------------------------------------
# Astronomy helpers (RA/Dec -> Alt/Az)
# ------------------------------------------------------------
def julian_date(dt_utc: datetime) -> float:
    y = dt_utc.year
    m = dt_utc.month
    d = dt_utc.day + (dt_utc.hour + dt_utc.minute / 60.0 + dt_utc.second / 3600.0) / 24.0
    if m <= 2:
        y -= 1
        m += 12
    A = y // 100
    B = 2 - A + (A // 4)
    JD = int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + B - 1524.5
    return JD


def gmst_deg(dt_utc: datetime) -> float:
    JD = julian_date(dt_utc)
    T = (JD - 2451545.0) / 36525.0
    gmst = (
        280.46061837
        + 360.98564736629 * (JD - 2451545.0)
        + 0.000387933 * T * T
        - (T * T * T) / 38710000.0
    )
    return gmst % 360.0

def alt_az_from_ra_dec(ra_hours: float, dec_deg: float, lat_deg: float, lon_deg: float, dt_utc: datetime):
    # clamp latitude to avoid invalid inputs and exact poles causing numeric issues
    lat_deg = max(-89.999999, min(89.999999, float(lat_deg)))
    lon_deg = float(lon_deg)

    lst_deg = (gmst_deg(dt_utc) + lon_deg) % 360.0
    ra_deg = (ra_hours * 15.0) % 360.0
    ha_deg = (lst_deg - ra_deg) % 360.0
    if ha_deg > 180:
        ha_deg -= 360.0

    ha  = math.radians(ha_deg)
    dec = math.radians(dec_deg)
    lat = math.radians(lat_deg)

    # altitude
    sin_alt = math.sin(dec) * math.sin(lat) + math.cos(dec) * math.cos(lat) * math.cos(ha)
    sin_alt = max(-1.0, min(1.0, sin_alt))
    alt = math.asin(sin_alt)

    # azimuth (robust atan2 form)
    y = math.sin(ha)
    x = math.cos(ha) * math.sin(lat) - math.tan(dec) * math.cos(lat)
    az = math.atan2(y, x)  # returns [-pi, pi]
    az_deg = (math.degrees(az) + 180.0) % 360.0  # normalize to [0, 360)

    return math.degrees(alt), az_deg


# ------------------------------------------------------------
# Same projection as sky.html (Alt/Az -> X/Y)
# (Used for /api/constellations so archive page matches topology)
# ------------------------------------------------------------
def project_altaz_to_xy(alt_deg: float, az_deg: float) -> tuple[float, float]:
    r = (90.0 - float(alt_deg)) * SKY_SCALE
    a = math.radians(float(az_deg))
    x = r * math.sin(a)
    y = -r * math.cos(a)
    return float(x), float(y)


def pick_constellation_audio(const_name: str) -> str | None:
    """
    One representative audio URL per constellation (for archive cards).
    Strategy: choose smallest magnitude among rows with audio_url.
    """
    best_url = None
    best_mag = 1e9
    for s in MAJOR_AUDIO:
        if (s.get("const_name") or "").strip() != const_name:
            continue
        url = s.get("audio")
        if not url:
            continue
        try:
            mag = float(s.get("mag", 99.0))
        except Exception:
            mag = 99.0
        if mag < best_mag:
            best_mag = mag
            best_url = url
    return best_url


# ------------------------------------------------------------
# Pages
# ------------------------------------------------------------
# app.py (only the route part needs to change)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/sonification")
def sonification():
    return render_template("sonification.html")


@app.route("/sky")
def sky_page():
    return render_template("sky.html")


CONSTELLATIONS_CSV = os.path.join(DATA_DIR, "constellations.csv")


def load_constellations_csv(abs_path: str):
    """
    Returns a dict keyed by normalized constellation name:
      {
        "orion": {
          "name": "ORION",
          "title": "...",
          "story": "...",
          "stars": [ {star fields...}, ... ]
        },
        ...
      }
    """
    if not os.path.exists(abs_path):
        print(f"[WARN] constellations.csv not found: {abs_path}")
        return {}

    df = pd.read_csv(abs_path)
    df.columns = [c.strip() for c in df.columns]
    df = df.where(pd.notnull(df), None)

    out = {}
    for _, r in df.iterrows():
        cname = (r.get("Constellation") or "").strip()
        if not cname:
            continue

        key = _norm_key(cname)
        title = (r.get("Mythic Title") or "").strip()
        story = (r.get("Full Story") or "").strip()

        star_name = (r.get("Star Name") or "").strip()
        ra = (r.get("RA") or "").strip()
        dec = (r.get("Dec") or "").strip()
        mag = r.get("Mag")
        try:
            mag = float(mag) if mag is not None else None
        except Exception:
            mag = None

        audio_file_cell = (r.get("Audio_File") or "").strip()
        resolved = resolve_audio_filename(audio_file_cell, star_name) if (audio_file_cell or star_name) else None

        audio_url = None
        audio_exists = False
        if resolved:
            abs_audio = os.path.join(AUDIO_DIR, resolved)
            audio_exists = os.path.exists(abs_audio)
            audio_url = f"/static/audio/{quote(resolved, safe='')}"

        if key not in out:
            out[key] = {
                "name": cname,  # keep as provided (e.g., ORION)
                "title": title,
                "story": story,
                "stars": [],
            }
        else:
            # IMPORTANT FIX:
            # Prefer the LONGEST story/title among rows (so full story wins over short summary)
            if title and len(title) > len(out[key].get("title") or ""):
                out[key]["title"] = title
            if story and len(story) > len(out[key].get("story") or ""):
                out[key]["story"] = story

        if star_name:
            out[key]["stars"].append(
                {
                    "name": star_name,
                    "ra": ra,
                    "dec": dec,
                    "mag": mag,
                    "audio": audio_url,
                    "audio_exists": bool(audio_exists),
                    "audio_file": resolved,
                }
            )

    print(f"[INFO] Loaded constellations.csv: {len(out)} constellations")
    return out


CONSTELLATION_STORIES = load_constellations_csv(CONSTELLATIONS_CSV)

@app.route("/constellations")
def constellations_page():
    # 1) Group ALL stars from star_map.csv (MAJOR_AUDIO) by constellation
    stars_by_key: dict[str, list[dict]] = {}
    for s in MAJOR_AUDIO:
        cname = (s.get("const_name") or "").strip()
        if not cname:
            continue
        k = _norm_key(cname)
        stars_by_key.setdefault(k, []).append(
            {
                "name": s.get("name") or "",
                # show RA/Dec in the modal (your template expects ra/dec strings)
                "ra": (s.get("meta") or {}).get("RA (hh:mm:ss)") or (s.get("meta") or {}).get("RA") or "",
                "dec": (s.get("meta") or {}).get("Dec (dd:mm:ss)") or (s.get("meta") or {}).get("Dec") or "",
                "mag": float(s.get("mag")) if s.get("mag") is not None else None,
                "audio": s.get("audio"),
                "audio_exists": bool(s.get("audio_exists", False)),
                "audio_file": s.get("audio_file"),
            }
        )

    # Optional: sort stars by magnitude (bright first)
    for k, arr in stars_by_key.items():
        arr.sort(key=lambda x: (x["mag"] is None, x["mag"] if x["mag"] is not None else 999, x["name"]))

    # 2) Build union of constellations:
    #    - those that have stories (CONSTELLATION_STORIES from constellations.csv)
    #    - those that exist in star_map.csv (MAJOR_AUDIO)
    all_keys = set(CONSTELLATION_STORIES.keys()) | set(stars_by_key.keys())

    items = []
    for k in sorted(all_keys):
        story_block = CONSTELLATION_STORIES.get(k, {})  # from constellations.csv
        stars = stars_by_key.get(k, [])                 # from star_map.csv

        # Name/title/story prefer story CSV, otherwise fallback to pretty name from stars
        fallback_name = ""
        if stars:
            # try to recover original constellation name from MAJOR_AUDIO
            # (grab first star's constellation via reverse lookup)
            for s in MAJOR_AUDIO:
                if _norm_key((s.get("const_name") or "").strip()) == k:
                    fallback_name = (s.get("const_name") or "").strip()
                    break

        name = (story_block.get("name") or fallback_name or "").strip()
        title = (story_block.get("title") or "").strip()
        story = (story_block.get("story") or "").strip()

        # Representative audio for the card button (brightest star with audio)
        rep_audio = None
        rep_mag = 1e9
        for st in stars:
            if not st.get("audio"):
                continue
            m = st.get("mag")
            m = float(m) if isinstance(m, (int, float)) else 99.0
            if m < rep_mag:
                rep_mag = m
                rep_audio = st.get("audio")

        items.append(
            {
                "name": name,
                "title": title,
                "story": story,
                "audio": rep_audio,
                "stars": stars,
            }
        )

    # If you want: hide constellations with no stars and no story
    items = [it for it in items if (it["name"] or it["story"] or it["stars"])]

    items.sort(key=lambda x: (x.get("name") or "").upper())
    return render_template("constellations.html", preload_data=items)



# ------------------------------------------------------------
# API: SKY (your existing interactive sky)
# ------------------------------------------------------------
@app.route("/api/sky")
def api_sky():
    lat = float(request.args.get("lat", 37.5665))
    lon = float(request.args.get("lon", 126.9780))
    ts = request.args.get("ts", "")
    min_alt = float(request.args.get("min_alt", -8.0))

    dt = datetime.fromisoformat(ts.replace("Z", "+00:00")) if ts else datetime.now(timezone.utc)

    # Audio stars grouped by CSV constellation name
    audio_by_const: dict[str, list[dict]] = {}
    for s in MAJOR_AUDIO:
        const_name = (s.get("const_name") or "Unknown").strip()
        if not const_name or _norm_key(const_name) not in CSV_CONST_KEYS:
            continue

        ra_h = (float(s["ra_deg"]) / 15.0) % 24.0
        alt, az = alt_az_from_ra_dec(ra_h, float(s["dec_deg"]), lat, lon, dt)
        if alt < min_alt:
            continue

        audio_by_const.setdefault(const_name, []).append(
            {
                "id": slugify(f"{const_name}-{s['name']}-{s['ra_deg']}-{s['dec_deg']}"),
                "name": s["name"],
                "const": const_name,
                "mag": float(s.get("mag", 3.0)),
                "alt": float(alt),
                "az": float(az),
                "audio": s.get("audio"),
                "audio_exists": bool(s.get("audio_exists", False)),
                "audio_file": s.get("audio_file"),
                "meta": s.get("meta") or {},
            }
        )

    # Constellation polylines + vertices as line_stars
    constellations = []
    for abbr, polylines in CONST_LINES.items():
        name = ABBR_TO_NAME.get(abbr, abbr)

        # Only constellations existing in CSV
        if _norm_key(name) not in CSV_CONST_KEYS:
            continue

        # IMPORTANT: HIDE constellations with 0 audio (after altitude filtering)
        audio_stars = audio_by_const.get(name, [])
        if not audio_stars:
            continue

        poly_altaz = []
        line_star_points = []

        for line in polylines:
            out_line = []
            for (ra_deg, dec_deg) in line:
                ra_h = (float(ra_deg) / 15.0) % 24.0
                alt, az = alt_az_from_ra_dec(ra_h, float(dec_deg), lat, lon, dt)
                out_line.append({"alt": float(alt), "az": float(az)})

                if alt >= min_alt:
                    line_star_points.append((round(float(alt), 4), round(float(az), 4)))

            if len(out_line) >= 2:
                poly_altaz.append(out_line)

        seen = set()
        unique_line_stars = []
        for alt, az in line_star_points:
            key = (alt, az)
            if key in seen:
                continue
            seen.add(key)
            unique_line_stars.append({"alt": alt, "az": az})

        constellations.append(
            {
                "abbr": abbr,
                "name": name,
                "polylines": poly_altaz,
                "line_stars": unique_line_stars,
                "audio_stars": audio_stars,  # already guaranteed non-empty
            }
        )

    consts_with_lines = len([c for c in constellations if c["polylines"]])
    print(
        "[SKY]",
        f"CSV constellations={len(CSV_CONST_NAMES_SET)} | "
        f"Audio rows={len(MAJOR_AUDIO)} | "
        f"Sent constellations={len(constellations)} | "
        f"With lines visible={consts_with_lines} | "
        f"min_alt={min_alt}"
    )

    return jsonify(
        {
            "constellations": constellations,
            "ts_utc": dt.isoformat().replace("+00:00", "Z"),
            "counts": {
                "constellations_sent": len(constellations),
                "constellations_with_lines_visible": consts_with_lines,
                "csv_constellations_total": len(CSV_CONST_NAMES_SET),
                "audio_csv_rows": len(MAJOR_AUDIO),
            },
        }
    )
@app.route("/api/constellation_report")
def api_constellation_report():
    """
    FULL constellation list from CSV_CONST_NAMES_SET (star_map.csv).
    GPT decides show/hide + one sentence + reason.
    NO try/except: failures will raise and crash the request.
    """

    lat = float(request.args.get("lat", 37.5665))
    lon = float(request.args.get("lon", 126.9780))
    season = (request.args.get("season") or "winter").strip().lower()

    # signals (for GPT + debugging)
    min_alt = float(request.args.get("min_alt", 10.0))
    max_mag = float(request.args.get("max_mag", 4.5))
    min_stars = int(request.args.get("min_stars", 1))
    max_send = int(request.args.get("max_send", 9999))

    # season -> reference date (internal only)
    y = datetime.now(timezone.utc).year
    season_map = {
        "spring": (3, 21),
        "summer": (6, 21),
        "fall":   (9, 22),
        "autumn": (9, 22),
        "winter": (12, 21),
    }
    m, d = season_map.get(season, (12, 21))
    dt = datetime(y, m, d, 21, 0, 0, tzinfo=timezone.utc)
    season_label = season.capitalize() if season in ("spring", "summer", "fall", "winter") else season

    # 1) Full constellation list from your CSV-derived set
    csv_constellations = sorted([c for c in CSV_CONST_NAMES_SET if (c or "").strip()], key=lambda s: s.upper())

    # 2) Signals for every constellation (init)
    by_const = {
        cname: {
            "total_audio_rows": 0,
            "pass": 0,
            "best_alt": -999.0,
            "avg_alt_sum": 0.0,
            "best_mag": 99.0,
            "avg_mag_sum": 0.0,
        }
        for cname in csv_constellations
    }

    # Accumulate signals from MAJOR_AUDIO
    for s in MAJOR_AUDIO:
        cname = (s.get("const_name") or "").strip()
        if not cname or cname not in by_const:
            continue

        mag = float(s.get("mag", 99.0))

        ra_h = (float(s["ra_deg"]) / 15.0) % 24.0
        alt, az = alt_az_from_ra_dec(ra_h, float(s["dec_deg"]), lat, lon, dt)

        info = by_const[cname]
        info["total_audio_rows"] += 1
        info["best_alt"] = max(info["best_alt"], float(alt))
        info["best_mag"] = min(info["best_mag"], float(mag))
        info["avg_alt_sum"] += float(alt)
        info["avg_mag_sum"] += float(mag)

        if (float(alt) >= min_alt) and (float(mag) <= max_mag):
            info["pass"] += 1

    # Finalize averages + build candidates
    candidates = []
    for cname in csv_constellations:
        info = by_const[cname]
        n = max(1, int(info["total_audio_rows"]))
        avg_alt = float(info["avg_alt_sum"]) / n
        avg_mag = float(info["avg_mag_sum"]) / n

        candidates.append(
            {
                "name": cname,
                "total_audio_rows": int(info["total_audio_rows"]),
                "pass": int(info["pass"]),
                "best_alt": round(float(info["best_alt"]), 2),
                "avg_alt": round(float(avg_alt), 2),
                "best_mag": round(float(info["best_mag"]), 2),
                "avg_mag": round(float(avg_mag), 2),
            }
        )

    # Sort: strongest first
    candidates.sort(key=lambda x: (-x["pass"], -x["best_alt"], x["best_mag"], x["name"].upper()))
    candidates_for_gpt = candidates[: max(10, min(max_send, len(candidates)))]

    # 3) GPT decides (NO try/except)
    if not _openai_client:
        raise RuntimeError("OpenAI client not configured (missing OPENAI_API_KEY or openai import failed).")

    sys = (
        "Return STRICT JSON only. "
        "You are an astronomy guide. "
        "Given a location (lat/lon), a season, and visibility signals per constellation, "
        "decide whether to SHOW or HIDE each constellation. "
        "For each constellation, write exactly ONE sentence (no time-of-day, no clock times). "
        "Also provide a short REASON for the show/hide decision."
    )

    payload = {
        "location": {"lat": lat, "lon": lon},
        "season": season_label,
        "signals_policy": {
            "min_alt_deg_signal": min_alt,
            "max_mag_signal": max_mag,
            "min_stars_signal": min_stars,
            "guidance": [
                "If pass >= min_stars_signal, prefer show.",
                "If pass == 0, usually hide.",
                "Use best_alt and pass as the strongest evidence.",
                "Keep reasons short and evidence-based (mention pass/best_alt/best_mag).",
                "Sentence must not contain any time or hour."
            ],
        },
        "constellations": candidates_for_gpt,
        "output_schema": {
            "items": [
                {
                    "name": "must match input name exactly",
                    "show": "boolean",
                    "sentence": "one sentence, no time mentioned",
                    "reason": "short reason referencing signals"
                }
            ]
        }
    }

    resp = _openai_client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": sys},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )

    raw = getattr(resp, "output_text", None) or str(resp)

    # NO try/except: if model returns non-JSON this will raise
    parsed = json.loads(raw)

    items = parsed["items"]  # will raise if missing / wrong type

    gpt_items_map = {}
    for it in items:
        nm = (it.get("name") or "").strip()
        if not nm:
            continue
        gpt_items_map[nm] = {
            "show": bool(it.get("show", False)),
            "sentence": (it.get("sentence") or "").strip(),
            "reason": (it.get("reason") or "").strip(),
        }

    # 4) Output for ALL csv_constellations
    items_out = []
    for cname in csv_constellations:
        sig = by_const[cname]
        passing = int(sig["pass"])
        total = int(sig["total_audio_rows"])

        if cname in gpt_items_map:
            show = bool(gpt_items_map[cname]["show"])
            sentence = gpt_items_map[cname]["sentence"]
            reason = gpt_items_map[cname]["reason"]
            gpt_decided = True
        else:
            # Fallback for constellations not included due to max_send
            show = passing >= min_stars
            sentence = f"{cname} is {'recommended' if show else 'not recommended'} in {season_label} from your location."
            reason = f"Fallback: pass={passing} (min_stars={min_stars}), best_alt={sig['best_alt']:.1f}, best_mag={sig['best_mag']:.1f}."
            gpt_decided = False

        items_out.append(
            {
                "name": cname,
                "show": show,
                "sentence": sentence,
                "reason": reason,
                "signals": {
                    "pass": passing,
                    "total_audio_rows": total,
                    "best_alt": float(sig["best_alt"]),
                    "best_mag": float(sig["best_mag"]),
                    "min_alt": min_alt,
                    "max_mag": max_mag,
                    "min_stars": min_stars,
                },
                "gpt_decided": gpt_decided,
            }
        )

    items_out.sort(
        key=lambda x: (
            not x["show"],
            -x["signals"]["pass"],
            -x["signals"]["best_alt"],
            x["signals"]["best_mag"],
            x["name"].upper(),
        )
    )

    shown_count = sum(1 for it in items_out if it["show"])
    gpt_decided_count = sum(1 for it in items_out if it["gpt_decided"])

    print(
        "[CONST_REPORT]",
        f"lat={lat:.4f} lon={lon:.4f} season={season_label} | "
        f"csv_total={len(csv_constellations)} shown={shown_count} | "
        f"sent_to_gpt={len(candidates_for_gpt)} gpt_decided={gpt_decided_count}"
    )

    return jsonify(
        {
            "season": season_label,
            "ts_utc": dt.isoformat().replace("+00:00", "Z"),
            "csv_constellations": csv_constellations,
            "items": items_out,
            "counts": {
                "csv_constellations_total": len(csv_constellations),
                "shown": shown_count,
                "gpt_decided": gpt_decided_count,
            },
            "gpt_used": True,
            "model": OPENAI_MODEL,
        }
    )

@app.route("/api/constellations")
def api_constellations():
    """
    Query params (optional):
      lat, lon, ts (ISO string), min_alt
    Defaults:
      min_alt = -90  (show full stick figures even below horizon)
    """
    lat = float(request.args.get("lat", 37.5665))
    lon = float(request.args.get("lon", 126.9780))
    ts = request.args.get("ts", "")
    min_alt = float(request.args.get("min_alt", -90.0))

    dt = datetime.fromisoformat(ts.replace("Z", "+00:00")) if ts else datetime.now(timezone.utc)

    out = []
    for abbr, polylines in CONST_LINES.items():
        name = ABBR_TO_NAME.get(abbr, abbr)

        if _norm_key(name) not in CSV_CONST_KEYS:
            continue

        poly_xy: list[list[list[float]]] = []
        verts_xy: list[tuple[float, float]] = []

        for line in polylines:
            xy_line: list[list[float]] = []
            for (ra_deg, dec_deg) in line:
                ra_h = (float(ra_deg) / 15.0) % 24.0
                alt, az = alt_az_from_ra_dec(ra_h, float(dec_deg), lat, lon, dt)

                if float(alt) < min_alt:
                    continue

                x, y = project_altaz_to_xy(alt, az)
                xy_line.append([x, y])
                verts_xy.append((round(x, 4), round(y, 4)))

            if len(xy_line) >= 2:
                poly_xy.append(xy_line)

        # unique vertex points
        seen = set()
        line_stars_xy: list[list[float]] = []
        for x, y in verts_xy:
            k = (x, y)
            if k in seen:
                continue
            seen.add(k)
            line_stars_xy.append([float(x), float(y)])

        out.append(
            {
                "abbr": abbr,
                "name": name,
                "audio": pick_constellation_audio(name),
                "polylines": poly_xy,
                "line_stars": line_stars_xy,
            }
        )

    out.sort(key=lambda c: (c.get("name") or ""))

    return jsonify(
        {
            "constellations": out,
            "ts_utc": dt.isoformat().replace("+00:00", "Z"),
            "counts": {
                "returned": len(out),
                "with_polylines": sum(1 for c in out if c.get("polylines")),
            },
        }
    )

@app.get("/api/audio_files")
def api_audio_files():
    audio_dir = os.path.join(app.static_folder, "audio")  # static/audio
    exts = {".mp3", ".wav", ".ogg", ".m4a", ".aac", ".flac"}

    items = []
    if os.path.isdir(audio_dir):
        for name in sorted(os.listdir(audio_dir)):
            path = os.path.join(audio_dir, name)
            _, ext = os.path.splitext(name.lower())
            if os.path.isfile(path) and ext in exts:
                items.append({
                    "id": name,                      # unique enough
                    "file": name,
                    "url": url_for("static", filename=f"audio/{name}")
                })

    return jsonify({"items": items})

@app.route("/api/audio_stars")
def api_audio_stars():
    """
    Directory data for the 'Audio Stars' section on index.html.
    Uses MAJOR_AUDIO (parsed from star_map.csv).
    """
    items = []
    for s in MAJOR_AUDIO:
        meta = s.get("meta") or {}
        # preserve human-readable RA/Dec strings for UI
        ra_str = meta.get("RA (hh:mm:ss)") or meta.get("RA") or ""
        dec_str = meta.get("Dec (dd:mm:ss)") or meta.get("Dec") or ""

        const_name = (s.get("const_name") or "Unknown").strip()
        name = (s.get("name") or "").strip()
        if not name:
            continue

        items.append(
            {
                "id": slugify(f"{const_name}-{name}-{s.get('ra_deg')}-{s.get('dec_deg')}"),
                "name": name,
                "const": const_name,
                "ra": str(ra_str),
                "dec": str(dec_str),
                "mag": float(s.get("mag")) if s.get("mag") is not None else None,
                "audio": s.get("audio"),
                "audio_file": s.get("audio_file"),
                "audio_exists": bool(s.get("audio_exists", False)),
            }
        )

    # default sort: brightest first (lower magnitude)
    items.sort(key=lambda x: (x["mag"] is None, x["mag"] if x["mag"] is not None else 999, x["name"]))

    return jsonify(
        {
            "items": items,
            "counts": {
                "total": len(items),
                "with_audio": sum(1 for it in items if it.get("audio")),
                "missing_audio": sum(1 for it in items if not it.get("audio")),
            },
        }
    )

# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
