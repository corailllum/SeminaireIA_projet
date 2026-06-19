"""
Agent Ollama (llama3.2) avec outils MCP
Utilise LangChain + LangGraph pour créer un agent ReAct capable d'interroger
les 3 sources de données (grimpeurs, grades, routes).

Note d'implémentation :
Les fonctions des serveurs MCP (mcp_climber.py, mcp_grades.py, mcp_routes.py)
sont importées et appelées directement en mémoire plutôt que via subprocess.
C'est plus rapide et plus fiable pour un agent local (pas de protocole JSON-RPC
à gérer, pas de risque de chemin relatif cassé). Les serveurs MCP restent
utilisables tels quels par un client MCP standard (Claude Desktop, etc.) si besoin.
"""

import json
from pathlib import Path

import pandas as pd
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain_core.messages import HumanMessage

try:
    # API récente (langchain >= 1.0)
    from langchain.agents import create_agent as create_react_agent
except ImportError:
    # API plus ancienne (langgraph.prebuilt)
    from langgraph.prebuilt import create_react_agent

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent / "Data"

# ─────────────────────────────────────────────
# Configuration Ollama
# ─────────────────────────────────────────────

OLLAMA_MODEL = "llama3.2"
OLLAMA_BASE_URL = "http://localhost:11434"

# ─────────────────────────────────────────────
# Chargement des données (une seule fois)
# ─────────────────────────────────────────────

df_climbers = pd.read_csv(DATA_DIR / "climber_df.csv")
df_grades = pd.read_csv(DATA_DIR / "grades_conversion_table.csv", index_col=0)
df_routes = pd.read_csv(DATA_DIR / "route_geo.csv", index_col=0)


# ─────────────────────────────────────────────
# Outils LangChain (logique directe, sans subprocess)
# ─────────────────────────────────────────────

@tool
def climber_stats() -> str:
    """Retourne les statistiques globales sur les grimpeurs : nombre total, âge moyen, grade moyen, répartition par sexe et par pays."""
    stats = {
        "total_grimpeurs": int(len(df_climbers)),
        "age_moyen": round(df_climbers["age"].mean(), 2),
        "grade_max_moyen": round(df_climbers["grades_max"].mean(), 2),
        "grade_mean_global": round(df_climbers["grades_mean"].mean(), 2),
        "repartition_sexe": df_climbers["sex"].value_counts().to_dict(),
        "pays_uniques": int(df_climbers["country"].nunique()),
        "top_pays": df_climbers["country"].value_counts().head(5).to_dict(),
    }
    return json.dumps(stats, ensure_ascii=False, indent=2)


@tool
def climbers_by_country(country: str) -> str:
    """Retourne les grimpeurs d'un pays donné avec leurs statistiques. Passer le code pays en majuscules (ex: SWE, NOR, GBR, USA, DEU)."""
    country = country.upper()
    filtered = df_climbers[df_climbers["country"] == country]
    if filtered.empty:
        return f"Aucun grimpeur trouvé pour le pays: {country}"
    result = {
        "country": country,
        "nombre": int(len(filtered)),
        "age_moyen": round(filtered["age"].mean(), 2),
        "grade_max_moyen": round(filtered["grades_max"].mean(), 2),
        "grimpeurs": filtered[["user_id", "age", "sex", "grades_max", "grades_mean"]].to_dict(orient="records"),
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


@tool
def top_climbers(n: int = 10) -> str:
    """Retourne les N meilleurs grimpeurs selon leur grade maximum. Par défaut N=10."""
    top = df_climbers.nlargest(n, "grades_max")[
        ["user_id", "country", "sex", "age", "grades_max", "grades_mean", "years_cl"]
    ]
    return top.to_json(orient="records", indent=2)


@tool
def climber_progression() -> str:
    """Analyse la progression des grimpeurs : différence entre leur premier et dernier grade enregistré."""
    df_copy = df_climbers.copy()
    df_copy["progression"] = df_copy["grades_last"] - df_copy["grades_first"]
    stats = {
        "progression_moyenne": round(df_copy["progression"].mean(), 2),
        "progression_max": int(df_copy["progression"].max()),
        "progression_min": int(df_copy["progression"].min()),
        "grimpeurs_en_progression": int((df_copy["progression"] > 0).sum()),
        "grimpeurs_en_regression": int((df_copy["progression"] < 0).sum()),
        "grimpeurs_stables": int((df_copy["progression"] == 0).sum()),
    }
    return json.dumps(stats, ensure_ascii=False, indent=2)


@tool
def grade_convert_id(grade_id: int) -> str:
    """Convertit un grade_id numérique en notation française (ex: grade_id=42 → '6a'). Utile pour interpréter les données numériques."""
    row = df_grades[df_grades["grade_id"] == grade_id]
    if row.empty:
        return f"grade_id {grade_id} introuvable."
    grade_fra = row["grade_fra"].values[0]
    return json.dumps({"grade_id": grade_id, "grade_fra": grade_fra})


@tool
def grade_convert_fra(grade_fra: str) -> str:
    """Convertit un grade français en grade_id numérique (ex: '7b+' → grade_id). Utile pour filtrer les données."""
    grade_fra = grade_fra.strip()
    row = df_grades[df_grades["grade_fra"] == grade_fra]
    if row.empty:
        return f"Grade français '{grade_fra}' introuvable."
    grade_id = int(row["grade_id"].values[0])
    return json.dumps({"grade_fra": grade_fra, "grade_id": grade_id})


@tool
def all_grades() -> str:
    """Retourne la table complète de conversion des grades d'escalade (numérique ↔ notation française)."""
    return json.dumps(df_grades.to_dict(orient="records"), ensure_ascii=False, indent=2)


@tool
def route_stats() -> str:
    """Retourne les statistiques globales sur les voies d'escalade : nombre total, grade moyen, répartition par pays et par cluster."""
    stats = {
        "total_voies": int(len(df_routes)),
        "grade_moyen_global": round(df_routes["grade_mean"].mean(), 2),
        "grade_max": round(df_routes["grade_mean"].max(), 2),
        "grade_min": round(df_routes["grade_mean"].min(), 2),
        "pays_uniques": int(df_routes["country"].nunique()),
        "top_pays": df_routes["country"].value_counts().head(5).to_dict(),
        "repartition_clusters": df_routes["cluster"].value_counts().to_dict(),
        "crags_uniques": int(df_routes["crag"].nunique()),
        "rating_moyen": round(df_routes["rating_tot"].mean(), 4),
    }
    return json.dumps(stats, ensure_ascii=False, indent=2)


@tool
def routes_by_country(country: str) -> str:
    """Retourne les voies d'escalade d'un pays donné. Utiliser le code pays en minuscules (ex: and, arg, fra, esp)."""
    country = country.lower()
    filtered = df_routes[df_routes["country"] == country]
    if filtered.empty:
        return f"Aucune voie trouvée pour le pays: {country}"
    result = {
        "country": country,
        "nombre_voies": int(len(filtered)),
        "grade_moyen": round(filtered["grade_mean"].mean(), 2),
        "crags": filtered["crag"].unique().tolist(),
        "voies": filtered[["name_id", "crag", "sector", "name", "grade_mean", "cluster", "rating_tot"]]
        .head(20)
        .to_dict(orient="records"),
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


@tool
def top_rated_routes(n: int = 10) -> str:
    """Retourne les N meilleures voies d'escalade selon leur rating. Par défaut N=10."""
    top = df_routes.nlargest(n, "rating_tot")[
        ["name_id", "country", "crag", "name", "grade_mean", "cluster", "rating_tot"]
    ]
    return top.to_json(orient="records", indent=2)


@tool
def routes_by_cluster(cluster: int) -> str:
    """Retourne les voies appartenant à un cluster donné (0, 1, 2 ou 3). Les clusters regroupent des voies aux caractéristiques similaires."""
    filtered = df_routes[df_routes["cluster"] == cluster]
    result = {
        "cluster": cluster,
        "nombre_voies": int(len(filtered)),
        "grade_moyen": round(filtered["grade_mean"].mean(), 2),
        "rating_moyen": round(filtered["rating_tot"].mean(), 4),
        "voies": filtered[["name_id", "country", "crag", "name", "grade_mean", "rating_tot"]]
        .head(20)
        .to_dict(orient="records"),
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


@tool
def routes_by_grade_range(min_grade: float, max_grade: float) -> str:
    """Retourne les voies dont le grade moyen est entre min_grade et max_grade (valeurs numériques)."""
    filtered = df_routes[(df_routes["grade_mean"] >= min_grade) & (df_routes["grade_mean"] <= max_grade)]
    result = {
        "plage_grade": f"{min_grade} - {max_grade}",
        "nombre_voies": int(len(filtered)),
        "voies": filtered[["name_id", "country", "crag", "name", "grade_mean", "cluster", "rating_tot"]]
        .head(20)
        .to_dict(orient="records"),
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────
# Construction de l'agent
# ─────────────────────────────────────────────

ALL_TOOLS = [
    climber_stats,
    climbers_by_country,
    top_climbers,
    climber_progression,
    grade_convert_id,
    grade_convert_fra,
    all_grades,
    route_stats,
    routes_by_country,
    top_rated_routes,
    routes_by_cluster,
    routes_by_grade_range,
]

SYSTEM_PROMPT = """Tu es un assistant expert en données d'escalade.
Tu as accès à trois sources de données via des outils MCP :
1. **Grimpeurs** : profils, performances, progression par pays
2. **Grades** : table de conversion entre grades numériques et notation française
3. **Voies** : caractéristiques des voies d'escalade par pays, crag et cluster

Réponds toujours en français. Utilise les outils pour répondre précisément aux questions.
Quand tu cites des grades numériques, propose leur équivalent en notation française si c'est utile.
"""


def build_agent():
    """Crée et retourne l'agent ReAct LangGraph."""
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,
    )
    try:
        # API récente (langchain.agents.create_agent)
        agent = create_react_agent(llm, ALL_TOOLS, system_prompt=SYSTEM_PROMPT)
    except TypeError:
        # API ancienne (langgraph.prebuilt.create_react_agent)
        agent = create_react_agent(llm, ALL_TOOLS, system_prompt=SYSTEM_PROMPT)
    return agent


def ask_agent(question: str, agent=None) -> str:
    """
    Pose une question à l'agent et retourne sa réponse.
    Si agent=None, en crée un nouveau.
    """
    if agent is None:
        agent = build_agent()

    result = agent.invoke({"messages": [HumanMessage(content=question)]})
    messages = result.get("messages", [])
    if messages:
        return messages[-1].content
    return "Pas de réponse."


# ─────────────────────────────────────────────
# Test rapide en ligne de commande
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Agent Escalade (llama3.2 + MCP) ===\n")
    agent = build_agent()

    questions = [
        "Combien y a-t-il de grimpeurs dans la base de données ?",
        "Quels sont les 3 meilleurs grimpeurs par grade maximum ?",
        "Donne-moi les statistiques globales sur les voies d'escalade.",
    ]

    for q in questions:
        print(f"Q: {q}")
        reponse = ask_agent(q, agent)
        print(f"R: {reponse}\n{'-'*60}\n")