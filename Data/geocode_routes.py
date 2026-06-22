"""
Géocodage des voies d'escalade
Ajoute les colonnes latitude/longitude à route.csv via Nominatim (OpenStreetMap).

- Les voies d'un même crag partagent les mêmes coordonnées (1 requête par crag)
- Respect du rate limit Nominatim : 1 requête/seconde
- Les échecs sont retentés avec des termes de recherche plus larges
- Un cache JSON évite de refaire les requêtes si on relance le script

Usage :
    python geocode_routes.py
    python geocode_routes.py --input route.csv --output route_geo.csv
"""

import argparse
import json
import time
import os
import pandas as pd
import urllib.request
import urllib.parse
from typing import Optional

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
HEADERS = {"User-Agent": "climbing-route-geocoder/1.0"}
RATE_LIMIT_SECONDS = 1.1       # Nominatim impose 1 req/s
CACHE_FILE = "geocode_cache.json"

# Correspondance codes pays courts → noms complets pour améliorer le géocodage
COUNTRY_MAP = {
    "and": "Andorra",
    "arg": "Argentina",
    "aus": "Australia",
    "aut": "Austria",
    "bel": "Belgium",
    "bra": "Brazil",
    "can": "Canada",
    "che": "Switzerland",
    "cze": "Czech Republic",
    "deu": "Germany",
    "dnk": "Denmark",
    "esp": "Spain",
    "fin": "Finland",
    "fra": "France",
    "gbr": "United Kingdom",
    "grc": "Greece",
    "hun": "Hungary",
    "irl": "Ireland",
    "ita": "Italy",
    "jpn": "Japan",
    "mex": "Mexico",
    "nld": "Netherlands",
    "nor": "Norway",
    "nzl": "New Zealand",
    "pol": "Poland",
    "prt": "Portugal",
    "rou": "Romania",
    "rus": "Russia",
    "slk": "Slovakia",
    "slv": "Slovenia",
    "swe": "Sweden",
    "tur": "Turkey",
    "usa": "United States",
    "zaf": "South Africa",
    # fallback : on garde le code tel quel s'il n'est pas dans la liste
}


# ─────────────────────────────────────────────
# Cache JSON
# ─────────────────────────────────────────────

def load_cache(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────
# Requête Nominatim
# ─────────────────────────────────────────────

def nominatim_search(query: str) -> Optional[tuple[float, float]]:
    """
    Envoie une requête à Nominatim et retourne (lat, lon) ou None.
    """
    params = urllib.parse.urlencode({
        "q": query,
        "format": "json",
        "limit": 1,
    })
    url = f"{NOMINATIM_URL}?{params}"
    req = urllib.request.Request(url, headers=HEADERS)

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            if data:
                return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception as e:
        print(f"    ⚠ Erreur réseau: {e}")
    return None


def geocode_crag(crag: str, country_code: str) -> Optional[tuple[float, float]]:
    """
    Tente de géocoder un crag avec plusieurs stratégies de recherche progressives.
    Retourne (lat, lon) ou None si tout échoue.
    """
    country_name = COUNTRY_MAP.get(country_code.lower(), country_code)

    # Stratégies du plus précis au plus large
    queries = [
        f"{crag} climbing {country_name}",
        f"{crag} escalade {country_name}",
        f"{crag} {country_name}",
        f"{crag}",
    ]

    for query in queries:
        result = nominatim_search(query)
        time.sleep(RATE_LIMIT_SECONDS)
        if result:
            return result

    return None


# ─────────────────────────────────────────────
# Traitement principal
# ─────────────────────────────────────────────

def geocode_routes(input_csv: str, output_csv: str):
    print(f"\n{'='*55}")
    print("  Géocodage des voies d'escalade")
    print(f"{'='*55}")
    print(f"  Entrée : {input_csv}")
    print(f"  Sortie : {output_csv}")
    print(f"  Cache  : {CACHE_FILE}\n")

    # Chargement
    df = pd.read_csv(input_csv, index_col=0)
    print(f"  {len(df)} voies chargées, {df['crag'].nunique()} crags uniques.\n")

    # Cache
    cache = load_cache(CACHE_FILE)
    print(f"  {len(cache)} entrées en cache.\n")

    # On construit la liste des crags uniques (crag + country)
    crags = df[["crag", "country"]].drop_duplicates().reset_index(drop=True)
    total_crags = len(crags)

    results = {}  # crag_key → (lat, lon) ou (None, None)
    found = 0
    failed = []

    for i, row in crags.iterrows():
        crag = row["crag"]
        country = row["country"]
        cache_key = f"{crag}|{country}"

        # Déjà en cache ?
        if cache_key in cache:
            coords = cache[cache_key]
            lat = coords.get("lat")
            lon = coords.get("lon")
            status = "✓ cache" if lat is not None else "✗ cache (échec précédent)"
        else:
            print(f"  [{i+1}/{total_crags}] Géocodage : {crag} ({country})...")
            coords_result = geocode_crag(crag, country)

            if coords_result:
                lat, lon = coords_result
                cache[cache_key] = {"lat": lat, "lon": lon}
                status = f"✓ ({lat:.4f}, {lon:.4f})"
            else:
                lat, lon = None, None
                cache[cache_key] = {"lat": None, "lon": None}
                failed.append(f"{crag} ({country})")
                status = "✗ non trouvé"

            save_cache(cache, CACHE_FILE)

        if lat is not None:
            found += 1
        else:
            failed.append(f"{crag} ({country})")

        results[cache_key] = (lat, lon)
        print(f"  [{i+1}/{total_crags}] {crag} ({country}) → {status}")

    # Ajout des colonnes au DataFrame
    df["latitude"] = df.apply(
        lambda r: results.get(f"{r['crag']}|{r['country']}", (None, None))[0], axis=1
    )
    df["longitude"] = df.apply(
        lambda r: results.get(f"{r['crag']}|{r['country']}", (None, None))[1], axis=1
    )

    # Sauvegarde
    df.to_csv(output_csv)

    # Résumé
    print(f"\n{'='*55}")
    print(f"  Résultats :")
    print(f"  ✓ {found}/{total_crags} crags géocodés avec succès")
    print(f"  ✗ {total_crags - found} crags non trouvés")
    print(f"  Voies avec coordonnées : {df['latitude'].notna().sum()}/{len(df)}")
    print(f"  Fichier sauvegardé    : {output_csv}")

    if failed:
        # Dédoublonnage de la liste des échecs
        failed_unique = list(dict.fromkeys(failed))
        print(f"\n  Crags non géocodés ({len(failed_unique)}) :")
        for f in failed_unique:
            print(f"    - {f}")
        print("\n  Conseil : vérifier l'orthographe ou ajouter manuellement")
        print("  les coordonnées dans geocode_cache.json pour ces crags.")

    print(f"{'='*55}\n")
    return df


# ─────────────────────────────────────────────
# Utilitaire : correction manuelle du cache
# ─────────────────────────────────────────────

def patch_cache(crag: str, country: str, lat: float, lon: float):
    """
    Ajoute ou corrige manuellement des coordonnées dans le cache.
    Exemple d'utilisation :
        patch_cache("montserrat", "and", 41.5930, 1.8385)
    """
    cache = load_cache(CACHE_FILE)
    key = f"{crag}|{country}"
    cache[key] = {"lat": lat, "lon": lon}
    save_cache(cache, CACHE_FILE)
    print(f"Cache mis à jour : {key} → ({lat}, {lon})")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Géocodage des voies d'escalade via Nominatim")
    parser.add_argument("--input",  default="routes_rated.csv",     help="CSV d'entrée (défaut: routes_rated.csv)")
    parser.add_argument("--output", default="route_geo.csv", help="CSV de sortie (défaut: route_geo.csv)")
    parser.add_argument(
        "--patch", nargs=4,
        metavar=("CRAG", "COUNTRY", "LAT", "LON"),
        help="Correction manuelle : --patch montserrat and 41.593 1.838"
    )
    args = parser.parse_args()

    if args.patch:
        crag, country, lat, lon = args.patch
        patch_cache(crag, country, float(lat), float(lon))
    else:
        geocode_routes(args.input, args.output)
