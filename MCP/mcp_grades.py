"""
MCP Server - Grade Conversion
Expose des outils pour convertir et interroger les grades d'escalade (grade_conversion_table.csv)
"""

import json
import pandas as pd
from pathlib import Path
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Chargement des données (Data/ est un dossier voisin de MCP/)
CSV_PATH = Path(__file__).parent.parent / "Data" / "grades_conversion_table.csv"
df = pd.read_csv(CSV_PATH, index_col=0)

app = Server("mcp-grades")


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="get_all_grades",
            description="Retourne la table complète de conversion des grades (grade_id → grade français).",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        Tool(
            name="convert_grade_id_to_fra",
            description="Convertit un grade_id numérique en grade français (ex: 5 → '1b').",
            inputSchema={
                "type": "object",
                "properties": {
                    "grade_id": {
                        "type": "integer",
                        "description": "L'identifiant numérique du grade",
                    }
                },
                "required": ["grade_id"],
            },
        ),
        Tool(
            name="convert_grade_fra_to_id",
            description="Convertit un grade français en grade_id numérique (ex: '6a' → 42).",
            inputSchema={
                "type": "object",
                "properties": {
                    "grade_fra": {
                        "type": "string",
                        "description": "Le grade en notation française (ex: 6a, 7b+, 8c)",
                    }
                },
                "required": ["grade_fra"],
            },
        ),
        Tool(
            name="get_grade_range",
            description="Retourne tous les grades dans une plage de grade_id (ex: de 30 à 50).",
            inputSchema={
                "type": "object",
                "properties": {
                    "min_id": {"type": "integer", "description": "Grade_id minimum"},
                    "max_id": {"type": "integer", "description": "Grade_id maximum"},
                },
                "required": ["min_id", "max_id"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:

    if name == "get_all_grades":
        result = df.to_dict(orient="records")
        return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

    elif name == "convert_grade_id_to_fra":
        grade_id = arguments.get("grade_id")
        row = df[df["grade_id"] == grade_id]
        if row.empty:
            return [TextContent(type="text", text=f"grade_id {grade_id} introuvable.")]
        grade_fra = row["grade_fra"].values[0]
        return [TextContent(type="text", text=json.dumps({"grade_id": grade_id, "grade_fra": grade_fra}))]

    elif name == "convert_grade_fra_to_id":
        grade_fra = arguments.get("grade_fra", "").strip()
        row = df[df["grade_fra"] == grade_fra]
        if row.empty:
            return [TextContent(type="text", text=f"Grade français '{grade_fra}' introuvable.")]
        grade_id = int(row["grade_id"].values[0])
        return [TextContent(type="text", text=json.dumps({"grade_fra": grade_fra, "grade_id": grade_id}))]

    elif name == "get_grade_range":
        min_id = arguments.get("min_id", 0)
        max_id = arguments.get("max_id", 100)
        filtered = df[(df["grade_id"] >= min_id) & (df["grade_id"] <= max_id)]
        return [TextContent(type="text", text=filtered.to_json(orient="records", indent=2))]

    return [TextContent(type="text", text=f"Outil inconnu: {name}")]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())