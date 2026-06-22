"""
MCP Server - Routes d'escalade
Expose des outils pour interroger les données des voies d'escalade (route_geo.csv)
"""

import json
import math
import numpy as np
import pandas as pd
from pathlib import Path
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Chargement des données (Data/ est un dossier voisin de MCP/)
CSV_PATH = Path(__file__).parent.parent / "Data" / "route_geo.csv"
GRADES_CSV_PATH = Path(__file__).parent.parent / "Data" / "grades_conversion_table.csv"

df = pd.read_csv(CSV_PATH, index_col=0)

# Fusion avec la table de conversion des grades pour disposer d'un libellé
# français (grade_fra) directement dans les réponses, sans appel supplémentaire
# au serveur mcp_grades (utile pour les LLM qui peinent avec les chaînes d'outils
# trop longues, ex: gemma4:e4b).
try:
    grades_df = pd.read_csv(GRADES_CSV_PATH, index_col=0)
    grades_df = grades_df.loc[:, ~grades_df.columns.str.contains("^Unnamed")]
    df["grade_id_round"] = df["grade_mean"].round().fillna(0).astype(int)
    df = df.merge(grades_df[["grade_id", "grade_fra"]], left_on="grade_id_round", right_on="grade_id", how="left")
except (FileNotFoundError, KeyError):
    # Si la table de grades est introuvable ou mal formée, les tools fonctionnent
    # quand même, simplement sans le libellé grade_fra.
    pass

# Noms des colonnes GPS dans route_geo.csv.
# Si tes colonnes ont un nom différent (ex: "lat"/"lon"), change juste ces deux lignes.
LAT_COL = "latitude"
LON_COL = "longitude"

app = Server("mcp-routes")


# ---------------------------------------------------------------------------
# Helpers de conversion "type-safe"
# Les LLM locaux (ex: llama3.2) envoient parfois des nombres sous forme de
# chaînes de caractères ("10" au lieu de 10). Ces fonctions normalisent
# n'importe quelle entrée (str, int, float, None) vers le bon type Python.
# ---------------------------------------------------------------------------

def to_int(value, default=None):
    """Convertit une valeur (str, int, float) en int de manière sûre."""
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    try:
        return int(float(str(value).strip()))
    except (ValueError, TypeError):
        return default


def to_float(value, default=None):
    """Convertit une valeur (str, int, float) en float de manière sûre."""
    if value is None or value == "":
        return default
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip().replace(",", "."))
    except (ValueError, TypeError):
        return default


def to_str(value, default=""):
    """Convertit une valeur en chaîne propre (strip + lower)."""
    if value is None:
        return default
    return str(value).strip().lower()


def haversine_km(lat1, lon1, lat2_array, lon2_array):
    """Distance haversine (en km), vectorisée avec numpy pour aller vite sur tout le DataFrame."""
    lat1_r, lon1_r = np.radians(lat1), np.radians(lon1)
    lat2_r, lon2_r = np.radians(lat2_array), np.radians(lon2_array)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="get_route_stats",
            description="Retourne des statistiques globales sur les voies d'escalade (nombre, grade moyen, répartition par pays, clusters).",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        Tool(
            name="recommend_routes",
            description=(
                "Recommande des voies d'escalade adaptées à un niveau donné (débutant, intermédiaire, avancé). "
                "Les seuils de niveau sont calculés automatiquement à partir de la distribution des grades du "
                "dataset, puis les voies sont triées par popularité/note (rating_tot) au sein du niveau choisi. "
                "Idéal pour répondre à des questions comme 'quelle voie conseiller à un débutant'."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "level": {
                        "type": ["string", "null"],
                        "description": "Niveau du grimpeur : 'debutant', 'intermediaire' ou 'avance' (défaut: 'debutant')",
                    },
                    "country": {
                        "type": ["string", "null"],
                        "description": "Code pays optionnel pour filtrer (ex: and, arg, fra, esp...)",
                    },
                    "n": {
                        "type": ["integer", "string", "null"],
                        "description": "Nombre de voies à recommander (défaut: 5)",
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="get_routes_by_country",
            description="Retourne les voies filtrées par pays avec leurs informations.",
            inputSchema={
                "type": "object",
                "properties": {
                    "country": {
                        "type": ["string", "null"],
                        "description": "Code pays (ex: and, arg, fra, esp...)",
                    }
                },
                "required": ["country"],
            },
        ),
        Tool(
            name="get_top_rated_routes",
            description="Retourne les N meilleures voies selon leur rating_tot.",
            inputSchema={
                "type": "object",
                "properties": {
                    "n": {
                        "type": ["integer", "string", "null"],
                        "description": "Nombre de voies à retourner (défaut: 10)",
                    }
                },
                "required": [],
            },
        ),
        Tool(
            name="get_routes_by_crag",
            description="Retourne toutes les voies d'un site d'escalade (crag) spécifique.",
            inputSchema={
                "type": "object",
                "properties": {
                    "crag": {
                        "type": ["string", "null"],
                        "description": "Nom du site d'escalade (ex: montserrat, bariloche...)",
                    }
                },
                "required": ["crag"],
            },
        ),
        Tool(
            name="get_routes_by_cluster",
            description="Retourne les voies appartenant à un cluster spécifique (0, 1, 2, 3).",
            inputSchema={
                "type": "object",
                "properties": {
                    "cluster": {
                        "type": ["integer", "string", "null"],
                        "description": "Numéro du cluster (0, 1, 2 ou 3)",
                    }
                },
                "required": ["cluster"],
            },
        ),
        Tool(
            name="get_routes_by_grade_range",
            description="Retourne les voies dont le grade moyen est dans une plage donnée.",
            inputSchema={
                "type": "object",
                "properties": {
                    "min_grade": {"type": ["number", "string", "null"], "description": "Grade moyen minimum"},
                    "max_grade": {"type": ["number", "string", "null"], "description": "Grade moyen maximum"},
                },
                "required": ["min_grade", "max_grade"],
            },
        ),
        Tool(
            name="get_nearest_routes",
            description="Retourne les N voies d'escalade individuelles les plus proches d'un point GPS donné (latitude/longitude), triées par distance croissante.",
            inputSchema={
                "type": "object",
                "properties": {
                    "latitude": {"type": ["number", "string", "null"], "description": "Latitude du point de référence"},
                    "longitude": {"type": ["number", "string", "null"], "description": "Longitude du point de référence"},
                    "n": {"type": ["integer", "string", "null"], "description": "Nombre de voies à retourner (défaut: 10)"},
                },
                "required": ["latitude", "longitude"],
            },
        ),
        Tool(
            name="get_nearest_crags",
            description="Retourne les N falaises (crags) les plus proches d'un point GPS donné (latitude/longitude), regroupées avec leurs statistiques, triées par distance croissante.",
            inputSchema={
                "type": "object",
                "properties": {
                    "latitude": {"type": ["number", "string", "null"], "description": "Latitude du point de référence"},
                    "longitude": {"type": ["number", "string", "null"], "description": "Longitude du point de référence"},
                    "n": {"type": ["integer", "string", "null"], "description": "Nombre de falaises à retourner (défaut: 5)"},
                },
                "required": ["latitude", "longitude"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    arguments = arguments or {}

    if name == "get_route_stats":
        stats = {
            "total_voies": int(len(df)),
            "grade_moyen_global": round(df["grade_mean"].mean(), 2),
            "grade_max": round(df["grade_mean"].max(), 2),
            "grade_min": round(df["grade_mean"].min(), 2),
            "pays_uniques": int(df["country"].nunique()),
            "top_pays": df["country"].value_counts().head(5).to_dict(),
            "repartition_clusters": df["cluster"].value_counts().to_dict(),
            "crags_uniques": int(df["crag"].nunique()),
            "rating_moyen": round(df["rating_tot"].mean(), 4),
        }
        return [TextContent(type="text", text=json.dumps(stats, ensure_ascii=False, indent=2))]

    elif name == "recommend_routes":
        level = to_str(arguments.get("level"), default="debutant").lower()
        country = to_str(arguments.get("country")).lower()
        n = to_int(arguments.get("n"), default=5)

        # Seuils calculés dynamiquement par tertiles de grade_mean, plus robuste
        # qu'un seuil fixe codé en dur qui dépendrait de l'échelle exacte du dataset.
        q1 = df["grade_mean"].quantile(1 / 3)
        q2 = df["grade_mean"].quantile(2 / 3)

        if level in ("debutant", "débutant", "beginner", "facile"):
            subset = df[df["grade_mean"] <= q1]
            niveau_label = "débutant"
        elif level in ("avance", "avancé", "expert", "difficile"):
            subset = df[df["grade_mean"] > q2]
            niveau_label = "avancé"
        else:
            subset = df[(df["grade_mean"] > q1) & (df["grade_mean"] <= q2)]
            niveau_label = "intermédiaire"

        if country:
            subset = subset[subset["country"] == country]

        if subset.empty:
            suffix = f" dans le pays '{country}'" if country else ""
            return [TextContent(
                type="text",
                text=f"Aucune voie trouvée pour le niveau '{niveau_label}'{suffix}."
            )]

        top = subset.sort_values("rating_tot", ascending=False).head(n)
        cols = ["name_id", "country", "crag", "name", "grade_mean", "rating_tot"]
        if "grade_fra" in top.columns:
            cols.append("grade_fra")

        result = {
            "niveau": niveau_label,
            "seuils_grade_mean": {
                "max_debutant": round(q1, 2),
                "max_intermediaire": round(q2, 2),
            },
            "nombre_voies_correspondantes": int(len(subset)),
            "voies_recommandees": top[cols].to_dict(orient="records"),
        }
        return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

    elif name == "get_routes_by_country":
        country = to_str(arguments.get("country"))
        filtered = df[df["country"] == country]
        if filtered.empty:
            return [TextContent(type="text", text=f"Aucune voie trouvée pour le pays: {country}")]
        result = {
            "country": country,
            "nombre_voies": int(len(filtered)),
            "grade_moyen": round(filtered["grade_mean"].mean(), 2),
            "crags": filtered["crag"].unique().tolist(),
            "voies": filtered[["name_id", "crag", "sector", "name", "grade_mean", "cluster", "rating_tot"]]
            .head(20)
            .to_dict(orient="records"),
        }
        return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

    elif name == "get_top_rated_routes":
        n = to_int(arguments.get("n"), default=10)
        top = df.nlargest(n, "rating_tot")[
            ["name_id", "country", "crag", "name", "grade_mean", "cluster", "rating_tot"]
        ]
        return [TextContent(type="text", text=top.to_json(orient="records", indent=2))]

    elif name == "get_routes_by_crag":
        crag = to_str(arguments.get("crag"))
        filtered = df[df["crag"] == crag]
        if filtered.empty:
            return [TextContent(type="text", text=f"Aucune voie trouvée pour le crag: {crag}")]
        return [TextContent(type="text", text=filtered.to_json(orient="records", indent=2))]

    elif name == "get_routes_by_cluster":
        cluster = to_int(arguments.get("cluster"))
        if cluster is None:
            return [TextContent(type="text", text="Paramètre 'cluster' invalide ou manquant.")]
        filtered = df[df["cluster"] == cluster]
        result = {
            "cluster": cluster,
            "nombre_voies": int(len(filtered)),
            "grade_moyen": round(filtered["grade_mean"].mean(), 2) if not filtered.empty else None,
            "rating_moyen": round(filtered["rating_tot"].mean(), 4) if not filtered.empty else None,
            "voies": filtered[["name_id", "country", "crag", "name", "grade_mean", "rating_tot"]]
            .head(20)
            .to_dict(orient="records"),
        }
        return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

    elif name == "get_routes_by_grade_range":
        min_g = to_float(arguments.get("min_grade"), default=0.0)
        max_g = to_float(arguments.get("max_grade"), default=100.0)
        filtered = df[(df["grade_mean"] >= min_g) & (df["grade_mean"] <= max_g)]
        result = {
            "plage_grade": f"{min_g} - {max_g}",
            "nombre_voies": int(len(filtered)),
            "voies": filtered[["name_id", "country", "crag", "name", "grade_mean", "cluster", "rating_tot"]]
            .head(20)
            .to_dict(orient="records"),
        }
        return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

    elif name == "get_nearest_routes":
        lat = to_float(arguments.get("latitude"))
        lon = to_float(arguments.get("longitude"))
        n = to_int(arguments.get("n"), default=10)

        if lat is None or lon is None:
            return [TextContent(type="text", text="Paramètres 'latitude'/'longitude' invalides ou manquants.")]
        if LAT_COL not in df.columns or LON_COL not in df.columns:
            return [TextContent(
                type="text",
                text=f"Colonnes GPS introuvables dans route_geo.csv (attendu: '{LAT_COL}'/'{LON_COL}')."
            )]

        valid = df.dropna(subset=[LAT_COL, LON_COL]).copy()
        valid["distance_km"] = haversine_km(lat, lon, valid[LAT_COL].values, valid[LON_COL].values)
        nearest = valid.nsmallest(n, "distance_km")[
            ["name_id", "country", "crag", "name", "grade_mean", "cluster", "rating_tot",
             LAT_COL, LON_COL, "distance_km"]
        ]
        nearest["distance_km"] = nearest["distance_km"].round(2)
        return [TextContent(type="text", text=nearest.to_json(orient="records", indent=2))]

    elif name == "get_nearest_crags":
        lat = to_float(arguments.get("latitude"))
        lon = to_float(arguments.get("longitude"))
        n = to_int(arguments.get("n"), default=5)

        if lat is None or lon is None:
            return [TextContent(type="text", text="Paramètres 'latitude'/'longitude' invalides ou manquants.")]
        if LAT_COL not in df.columns or LON_COL not in df.columns:
            return [TextContent(
                type="text",
                text=f"Colonnes GPS introuvables dans route_geo.csv (attendu: '{LAT_COL}'/'{LON_COL}')."
            )]

        valid = df.dropna(subset=[LAT_COL, LON_COL]).copy()

        # Regroupement par falaise : une falaise partage en général les mêmes coordonnées GPS
        crags = valid.groupby(["crag", "country"]).agg(
            latitude=(LAT_COL, "mean"),
            longitude=(LON_COL, "mean"),
            nombre_voies=("name_id", "count"),
            grade_moyen=("grade_mean", "mean"),
            rating_moyen=("rating_tot", "mean"),
        ).reset_index()

        crags["distance_km"] = haversine_km(lat, lon, crags["latitude"].values, crags["longitude"].values)
        nearest = crags.nsmallest(n, "distance_km").copy()
        nearest["distance_km"] = nearest["distance_km"].round(2)
        nearest["grade_moyen"] = nearest["grade_moyen"].round(2)
        nearest["rating_moyen"] = nearest["rating_moyen"].round(4)

        return [TextContent(type="text", text=nearest.to_json(orient="records", indent=2))]

    return [TextContent(type="text", text=f"Outil inconnu: {name}")]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())