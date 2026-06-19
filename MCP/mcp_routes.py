"""
MCP Server - Routes d'escalade
Expose des outils pour interroger les données des voies d'escalade (route.csv)
"""

import json
import pandas as pd
from pathlib import Path
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Chargement des données (Data/ est un dossier voisin de MCP/)
CSV_PATH = Path(__file__).parent.parent / "Data" / "route_geo.csv"
df = pd.read_csv(CSV_PATH, index_col=0)

app = Server("mcp-routes")


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="get_route_stats",
            description="Retourne des statistiques globales sur les voies d'escalade (nombre, grade moyen, répartition par pays, clusters).",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        Tool(
            name="get_routes_by_country",
            description="Retourne les voies filtrées par pays avec leurs informations.",
            inputSchema={
                "type": "object",
                "properties": {
                    "country": {
                        "type": "string",
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
                        "type": "integer",
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
                        "type": "string",
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
                        "type": "integer",
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
                    "min_grade": {"type": "number", "description": "Grade moyen minimum"},
                    "max_grade": {"type": "number", "description": "Grade moyen maximum"},
                },
                "required": ["min_grade", "max_grade"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:

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

    elif name == "get_routes_by_country":
        country = arguments.get("country", "").lower()
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
        n = arguments.get("n", 10)
        top = df.nlargest(n, "rating_tot")[
            ["name_id", "country", "crag", "name", "grade_mean", "cluster", "rating_tot"]
        ]
        return [TextContent(type="text", text=top.to_json(orient="records", indent=2))]

    elif name == "get_routes_by_crag":
        crag = arguments.get("crag", "").lower()
        filtered = df[df["crag"] == crag]
        if filtered.empty:
            return [TextContent(type="text", text=f"Aucune voie trouvée pour le crag: {crag}")]
        return [TextContent(type="text", text=filtered.to_json(orient="records", indent=2))]

    elif name == "get_routes_by_cluster":
        cluster = arguments.get("cluster")
        filtered = df[df["cluster"] == cluster]
        result = {
            "cluster": cluster,
            "nombre_voies": int(len(filtered)),
            "grade_moyen": round(filtered["grade_mean"].mean(), 2),
            "rating_moyen": round(filtered["rating_tot"].mean(), 4),
            "voies": filtered[["name_id", "country", "crag", "name", "grade_mean", "rating_tot"]]
            .head(20)
            .to_dict(orient="records"),
        }
        return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

    elif name == "get_routes_by_grade_range":
        min_g = arguments.get("min_grade", 0)
        max_g = arguments.get("max_grade", 100)
        filtered = df[(df["grade_mean"] >= min_g) & (df["grade_mean"] <= max_g)]
        result = {
            "plage_grade": f"{min_g} - {max_g}",
            "nombre_voies": int(len(filtered)),
            "voies": filtered[["name_id", "country", "crag", "name", "grade_mean", "cluster", "rating_tot"]]
            .head(20)
            .to_dict(orient="records"),
        }
        return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

    return [TextContent(type="text", text=f"Outil inconnu: {name}")]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())