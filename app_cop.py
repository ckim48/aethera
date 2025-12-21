# app.py
from flask import Flask, render_template, jsonify, request
import pandas as pd
import math
import os
import re
import json
from datetime import datetime, timezone
from urllib.parse import quote

app = Flask(__name__)

MAJOR_CSV = os.path.join(app.root_path, "static", "data", "star_map.csv")
AUDIO_DIR = os.path.join(app.root_path, "static", "audio")
CONST_LINES_JSON = os.path.join(app.root_path, "static", "data", "constellation_lines.json")

# =========================
# Prepared myth stories (optional)
# =========================
STORY_OVERRIDES = {
    "Orion": """ORION — The Hunter Who Challenged the Heavens
Orion’s story begins long before he was placed among the stars...""",
}

# =========================
# Helpers
# =========================
def slugify(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-") or "star"

def _norm_key(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s

def _norm_const_key(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

# =========================
# Load constellation templates (stars + edges)
# =========================
def _load_const_templates(path: str):
    if not os.path.exists(path):
        print(f"[WARN] constellation_lines.json not found: {path}")
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            print("[WARN] constellation_lines.json must be an object/dict")
            return {}
        out = {}
        for k, v in data.items():
            if not isinstance(v, dict):
                continue
            stars = v.get("stars") or []
            edges = v.get("edges") or []
            if not isinstance(stars, list) or not isinstance(edges, list):
                continue
            out[str(k).strip()] = {"stars": stars, "edges": edges}
        print(f"[INFO] Loaded constellation templates: {len(out)} from {path}")
        return out
    except Exception as e:
        print(f"[WARN] Failed to load constellation_lines.json: {e}")
        return {}

CONST_TEMPLATES = _load_const_templates(CONST_LINES_JSON)
CONST_TEMPLATES_NORM = {_norm_const_key(k): v for k, v in CONST_TEMPLATES.items()}

# =========================
# Audio index
# =========================
def _build_audio_index(audio_dir: str):
    idx = []
    if not os.path.isdir(audio_dir):
        print(f"[WARN] Audio folder missing: {audio_dir}")
        return idx

    for fn in os.listdir(audio_dir):
        if not fn.lower().endswith(".mp3"):
            continue
        base = os.path.splitext(fn)[0]
        idx.append({
            "fn": fn,
            "norm": _norm_key(base),
            "url": f"/static/audio/{quote(fn)}"
        })

    idx.sort(key=lambda x: len(x["norm"]))
    print(f"[INFO] Indexed audio files: {len(idx)} from {audio_dir}")
    return idx

AUDIO_INDEX = _build_audio_index(AUDIO_DIR)

def find_audio_for_star(star_name: str):
    key = _norm_key(star_name)
    if not key:
        return None
    for it in AUDIO_INDEX:
        if key in it["norm"]:
            return it["url"]
    return None

# =========================
# Parsing RA/Dec
# =========================
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

# =========================
# Astronomy helpers
# =========================
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
    lst_deg = (gmst_deg(dt_utc) + lon_deg) % 360.0
    ra_deg = (ra_hours * 15.0) % 360.0
    ha_deg = (lst_deg - ra_deg) % 360.0
    if ha_deg > 180:
        ha_deg -= 360

    ha = math.radians(ha_deg)
    dec = math.radians(dec_deg)
    lat = math.radians(lat_deg)

    sin_alt = math.sin(dec) * math.sin(lat) + math.cos(dec) * math.cos(lat) * math.cos(ha)
    sin_alt = max(-1.0, min(1.0, sin_alt))
    alt = math.asin(sin_alt)

    cos_az = (math.sin(dec) - math.sin(alt) * math.sin(lat)) / (math.cos(alt) * math.cos(lat) + 1e-12)
    cos_az = max(-1.0, min(1.0, cos_az))
    az = math.acos(cos_az)
    if math.sin(ha) > 0:
        az = 2 * math.pi - az

    return math.degrees(alt), math.degrees(az)

def project_az_alt(az_deg: float, alt_deg: float, scale: float):
    r = (90.0 - alt_deg) * scale
    az = math.radians(az_deg)
    x = r * math.sin(az)
    y = -r * math.cos(az)
    return x, y

# =========================
# CSV load (audio stars only)
# =========================
def load_major_csv(abs_path: str):
    if not os.path.exists(abs_path):
        print(f"[WARN] CSV not found: {abs_path}")
        return []

    df = pd.read_csv(abs_path)
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

        audio_url = None
        file_cell = r.get("File")
        if file_cell:
            fn = str(file_cell).strip()
            if not fn.lower().endswith(".mp3"):
                fn += ".mp3"
            audio_url = f"/static/audio/{quote(fn)}"
        else:
            audio_url = find_audio_for_star(name)

        meta = r.to_dict()

        stars.append({
            "id": slugify(f"{const}-{name}"),
            "name": name,
            "const": const,
            "ra_h": float(ra_h),
            "dec_d": float(dec_d),
            "mag": float(mag),
            "audio": audio_url,
            "meta": meta,
            "is_main": True
        })

    return stars

MAJOR_DB = load_major_csv(MAJOR_CSV)
STARS_DB = list(MAJOR_DB)
print(f"[INFO] Loaded CSV audio stars: {len(MAJOR_DB)} from {MAJOR_CSV}")

# =========================
# Template edges only (IMPORTANT CHANGE)
# - We DO NOT generate fake stars from xy anymore.
# - We only build edges among stars that actually exist in your CSV.
# =========================
def augment_constellation_stars(stars_in_const):
    """
    returns:
      (same stars_in_const,
       lines_by_id: list[[idA,idB]] or None,
       lines_by_name: list[[nameA,nameB]] or None)
    """
    if not stars_in_const:
        return stars_in_const, None, None

    const = (stars_in_const[0].get("const") or "").strip() or "Unknown"
    tmpl = CONST_TEMPLATES_NORM.get(_norm_const_key(const))
    if not tmpl:
        return stars_in_const, None, None

    # build name->id for ONLY existing stars
    name_to_id = {}
    for s in stars_in_const:
        nm = _norm_key(s.get("name"))
        if nm:
            name_to_id[nm] = s.get("id")

    lines_by_id = []
    lines_by_name = []

    for a, b in (tmpl.get("edges") or []):
        a = (a or "").strip()
        b = (b or "").strip()
        if not a or not b or a == b:
            continue

        lines_by_name.append([a, b])

        ida = name_to_id.get(_norm_key(a))
        idb = name_to_id.get(_norm_key(b))
        if ida and idb and ida != idb:
            lines_by_id.append([ida, idb])

    return stars_in_const, (lines_by_id or None), (lines_by_name or None)

# =========================
# Pages
# =========================
@app.route("/")
def index():
    return render_template("sky.html")

@app.route("/constellations")
def constellations_page():
    return render_template("constellations.html")

# =========================
# APIs
# =========================
@app.route("/api/sky")
def api_sky():
    lat = float(request.args.get("lat", 37.5665))
    lon = float(request.args.get("lon", 126.9780))
    ts = request.args.get("ts", "")
    scale = float(request.args.get("scale", 55.0))
    min_alt = float(request.args.get("min_alt", -8.0))

    dt = datetime.fromisoformat(ts.replace("Z", "+00:00")) if ts else datetime.now(timezone.utc)

    groups = {}
    for s in STARS_DB:
        c = (s.get("const") or "").strip() or "Unknown"
        groups.setdefault(c, []).append(dict(s))

    const_lines_ids = {}
    const_lines_names = {}
    for c, members in groups.items():
        aug, lines_id, lines_name = augment_constellation_stars(members)
        groups[c] = aug
        if lines_id:
            const_lines_ids[c] = lines_id
        if lines_name:
            const_lines_names[c] = lines_name

    visible = []
    major_visible = 0

    for c, members in groups.items():
        for s in members:
            alt, az = alt_az_from_ra_dec(s["ra_h"], s["dec_d"], lat, lon, dt)
            if alt < min_alt:
                continue

            x, y = project_az_alt(az, alt, scale)
            out = dict(s)
            out["alt"] = float(alt)
            out["az"] = float(az)
            out["x"] = float(x)
            out["y"] = float(y)
            visible.append(out)
            if out.get("is_main"):
                major_visible += 1

    return jsonify({
        "stars": visible,
        "const_lines_ids": const_lines_ids,
        "const_lines_names": const_lines_names,
        "ts_utc": dt.isoformat().replace("+00:00", "Z"),
        "counts": {
            "loaded_major": len(MAJOR_DB),
            "visible_major": major_visible
        }
    })

@app.route("/api/constellations")
def api_constellations():
    groups = {}
    for s in STARS_DB:
        c = (s.get("const") or "").strip() or "Unknown"
        groups.setdefault(c, []).append(dict(s))

    out = []
    for c in sorted(groups.keys(), key=lambda x: x.lower()):
        members = groups[c]
        members, lines_id, lines_name = augment_constellation_stars(members)

        tracks = []
        for m in members:
            if m.get("audio"):
                meta = m.get("meta", {}) or {}
                tracks.append({
                    "star": m.get("name") or "Star",
                    "audio": m.get("audio"),
                    "emotion": (meta.get("Emotion Code") or "").strip() if meta.get("Emotion Code") else "",
                    "tempo": (meta.get("Tempo (BPM)") or "").strip() if meta.get("Tempo (BPM)") else "",
                    "instrument": (meta.get("Suggested Instrument Set") or "").strip() if meta.get("Suggested Instrument Set") else "",
                })

        constellation_audio = tracks[0]["audio"] if tracks else None
        story = STORY_OVERRIDES.get(c) or f"{c.upper()} — Myth story coming soon."

        out.append({
            "name": c,
            "story": story,
            "audio": constellation_audio,
            "track_count": len(tracks),
            "tracks": tracks,
            "lines_ids": lines_id or [],
            "lines_names": lines_name or [],
            "has_template": bool(CONST_TEMPLATES_NORM.get(_norm_const_key(c)))
        })

    return jsonify({"constellations": out})

if __name__ == "__main__":
    app.run(debug=True)
