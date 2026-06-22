"""
MCP Server - Climber Data
Expose des outils pour interroger les données des grimpeurs (climber_df.csv)
"""

import json
import pandas as pd
from pathlib import Path
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Chargement des données (Data/ est un dossier voisin de MCP/)
CSV_PATH = Path(__file__).parent.parent / "Data" / "climber_df.csv"
df = pd.read_csv(CSV_PATH)

app = Server("mcp-climber")


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


def to_str(value, default=""):
    """Convertit une valeur en chaîne propre (strip), sans forcer la casse."""
    if value is None:
        return default
    return str(value).strip()


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


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="get_climber_stats",
            description="Retourne des statistiques globales sur les grimpeurs (nombre total, âge moyen, grade moyen, répartition par sexe).",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        Tool(
            name="get_climbers_by_country",
            description="Retourne un échantillon de grimpeurs filtrés par pays avec leurs statistiques (échantillon limité à 20 pour rester léger ; utiliser get_top_climbers_by_country pour un classement précis).",
            inputSchema={
                "type": "object",
                "properties": {
                    "country": {
                        "type": ["string", "null"],
                        "description": "Code pays (ex: SWE, NOR, GBR, USA, DEU...)",
                    }
                },
                "required": ["country"],
            },
        ),
        Tool(
            name="get_top_climbers_by_country",
            description="Retourne les N meilleurs grimpeurs d'un pays donné, triés par grade maximum. Idéal pour répondre en un seul appel à des questions comme 'les 5 meilleurs grimpeurs français'.",
            inputSchema={
                "type": "object",
                "properties": {
                    "country": {
                        "type": ["string", "null"],
                        "description": "Code pays (ex: SWE, NOR, GBR, USA, DEU...)",
                    },
                    "n": {
                        "type": ["integer", "string", "null"],
                        "description": "Nombre de grimpeurs à retourner (défaut: 10)",
                    },
                },
                "required": ["country"],
            },
        ),
        Tool(
            name="search_climbers",
            description=(
                "Recherche des grimpeurs selon des critères physiques et de niveau. "
                "Tous les critères sont optionnels et peuvent être combinés librement : "
                "l'utilisateur peut n'en spécifier qu'un seul (ex: juste le poids) ou plusieurs "
                "(ex: taille + grade). Si aucun critère n'est fourni, retourne un échantillon "
                "non filtré. Les résultats sont triés par grade moyen décroissant."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "min_height": {"type": ["number", "string", "null"], "description": "Taille minimum en cm"},
                    "max_height": {"type": ["number", "string", "null"], "description": "Taille maximum en cm"},
                    "min_weight": {"type": ["number", "string", "null"], "description": "Poids minimum en kg"},
                    "max_weight": {"type": ["number", "string", "null"], "description": "Poids maximum en kg"},
                    "sex": {
                        "type": ["integer", "string", "null"],
                        "description": "Genre du grimpeur (0 = Homme, 1 = Femme)",
                    },
                    "min_grade": {"type": ["number", "string", "null"], "description": "Grade moyen minimum (grades_mean)"},
                    "max_grade": {"type": ["number", "string", "null"], "description": "Grade moyen maximum (grades_mean)"},
                    "n": {
                        "type": ["integer", "string", "null"],
                        "description": "Nombre de grimpeurs à retourner (défaut: 20)",
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="get_top_climbers",
            description="Retourne les N meilleurs grimpeurs selon leur grade maximum.",
            inputSchema={
                "type": "object",
                "properties": {
                    "n": {
                        "type": ["integer", "string", "null"],
                        "description": "Nombre de grimpeurs à retourner (défaut: 10)",
                    }
                },
                "required": [],
            },
        ),
        Tool(
            name="get_climber_by_id",
            description="Retourne les informations détaillées d'un grimpeur selon son user_id.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": ["integer", "string", "null"],
                        "description": "L'identifiant unique du grimpeur",
                    }
                },
                "required": ["user_id"],
            },
        ),
        Tool(
            name="get_progression_analysis",
            description="Analyse la progression des grimpeurs (différence entre grade_first et grade_last).",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    arguments = arguments or {}

    if name == "get_climber_stats":
        stats = {
            "total_grimpeurs": int(len(df)),
            "age_moyen": round(df["age"].mean(), 2),
            "grade_max_moyen": round(df["grades_max"].mean(), 2),
            "grade_mean_global": round(df["grades_mean"].mean(), 2),
            "repartition_sexe": df["sex"].value_counts().to_dict(),
            "pays_uniques": int(df["country"].nunique()),
            "top_pays": df["country"].value_counts().head(5).to_dict(),
        }
        return [TextContent(type="text", text=json.dumps(stats, ensure_ascii=False, indent=2))]

    elif name == "get_climbers_by_country":
        country = to_str(arguments.get("country")).upper()
        filtered = df[df["country"] == country]
        if filtered.empty:
            return [TextContent(type="text", text=f"Aucun grimpeur trouvé pour le pays: {country}")]
        result = {
            "country": country,
            "nombre": int(len(filtered)),
            "age_moyen": round(filtered["age"].mean(), 2),
            "grade_max_moyen": round(filtered["grades_max"].mean(), 2),
            # Limité à 20 pour éviter de saturer le contexte du LLM (un pays peut
            # contenir plusieurs centaines de grimpeurs). Utiliser get_top_climbers_by_country
            # pour un classement précis (top N triés par niveau).
            "grimpeurs_echantillon": filtered[["user_id", "age", "sex", "grades_max", "grades_mean"]]
            .head(20)
            .to_dict(orient="records"),
        }
        return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

    elif name == "get_top_climbers_by_country":
        country = to_str(arguments.get("country")).upper()
        n = to_int(arguments.get("n"), default=10)
        filtered = df[df["country"] == country]
        if filtered.empty:
            return [TextContent(type="text", text=f"Aucun grimpeur trouvé pour le pays: {country}")]
        top = filtered.nlargest(n, "grades_max")[
            ["user_id", "country", "sex", "age", "grades_max", "grades_mean", "years_cl"]
        ]
        return [TextContent(type="text", text=top.to_json(orient="records", indent=2))]

    elif name == "search_climbers":
        filtered = df.copy()
        criteres = {}

        min_height = to_float(arguments.get("min_height"))
        max_height = to_float(arguments.get("max_height"))
        min_weight = to_float(arguments.get("min_weight"))
        max_weight = to_float(arguments.get("max_weight"))
        sex = to_int(arguments.get("sex"))
        min_grade = to_float(arguments.get("min_grade"))
        max_grade = to_float(arguments.get("max_grade"))
        n = to_int(arguments.get("n"), default=20)

        if min_height is not None:
            filtered = filtered[filtered["height"] >= min_height]
            criteres["min_height"] = min_height
        if max_height is not None:
            filtered = filtered[filtered["height"] <= max_height]
            criteres["max_height"] = max_height
        if min_weight is not None:
            filtered = filtered[filtered["weight"] >= min_weight]
            criteres["min_weight"] = min_weight
        if max_weight is not None:
            filtered = filtered[filtered["weight"] <= max_weight]
            criteres["max_weight"] = max_weight
        if sex is not None:
            filtered = filtered[filtered["sex"] == sex]
            criteres["sex"] = sex
        if min_grade is not None:
            filtered = filtered[filtered["grades_mean"] >= min_grade]
            criteres["min_grade"] = min_grade
        if max_grade is not None:
            filtered = filtered[filtered["grades_mean"] <= max_grade]
            criteres["max_grade"] = max_grade

        if filtered.empty:
            return [TextContent(
                type="text",
                text=f"Aucun grimpeur trouvé pour les critères: {criteres}"
            )]

        result_df = filtered.sort_values("grades_mean", ascending=False).head(n)[
            ["user_id", "country", "sex", "age", "height", "weight", "grades_max", "grades_mean"]
        ]

        result = {
            "criteres_appliques": criteres,
            "nombre_total_correspondant": int(len(filtered)),
            "nombre_affiche": int(len(result_df)),
            "grimpeurs": result_df.to_dict(orient="records"),
        }
        return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

    elif name == "get_top_climbers":
        n = to_int(arguments.get("n"), default=10)
        top = df.nlargest(n, "grades_max")[
            ["user_id", "country", "sex", "age", "grades_max", "grades_mean", "years_cl"]
        ]
        return [TextContent(type="text", text=top.to_json(orient="records", indent=2))]

    elif name == "get_climber_by_id":
        user_id = to_int(arguments.get("user_id"))
        if user_id is None:
            return [TextContent(type="text", text="Paramètre 'user_id' invalide ou manquant.")]
        row = df[df["user_id"] == user_id]
        if row.empty:
            return [TextContent(type="text", text=f"Aucun grimpeur avec user_id={user_id}")]
        return [TextContent(type="text", text=row.to_json(orient="records", indent=2))]

    elif name == "get_progression_analysis":
        df_copy = df.copy()
        df_copy["progression"] = df_copy["grades_last"] - df_copy["grades_first"]
        stats = {
            "progression_moyenne": round(df_copy["progression"].mean(), 2),
            "progression_max": int(df_copy["progression"].max()),
            "progression_min": int(df_copy["progression"].min()),
            "grimpeurs_en_progression": int((df_copy["progression"] > 0).sum()),
            "grimpeurs_en_regression": int((df_copy["progression"] < 0).sum()),
            "grimpeurs_stables": int((df_copy["progression"] == 0).sum()),
        }
        return [TextContent(type="text", text=json.dumps(stats, ensure_ascii=False, indent=2))]

    return [TextContent(type="text", text=f"Outil inconnu: {name}")]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())