# app.py — Interface Streamlit avec sélecteur de modèle IA (Llama 3.2 vs Gemma)
import streamlit as st
import asyncio
import os
import json
import re
from pathlib import Path

import nest_asyncio
nest_asyncio.apply()

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
import pandas as pd
import plotly.express as px

# ─────────────────────────────────────────────
# Chemins et Chargement des Données
# ─────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent
MCP_DIR = PROJECT_ROOT / "MCP"
DATA_DIR = PROJECT_ROOT / "Data"

@st.cache_data
def load_data():
    climbers = pd.read_csv(DATA_DIR / "climber_df.csv")
    routes = pd.read_csv(DATA_DIR / "route_geo.csv")
    grades = pd.read_csv(DATA_DIR / "grades_conversion_table.csv")

    grades = grades.loc[:, ~grades.columns.str.contains('^Unnamed')]

    # Traduction des grades numériques
    routes["grade_id_round"] = routes["grade_mean"].round().fillna(0).astype(int)
    routes = routes.merge(grades, left_on="grade_id_round", right_on="grade_id", how="left")

    climbers["grade_id_round"] = climbers["grades_mean"].round().fillna(0).astype(int)
    climbers = climbers.merge(grades, left_on="grade_id_round", right_on="grade_id", how="left")

    return climbers, routes, grades

climbers_df, routes_df, grades_df = load_data()

# ─────────────────────────────────────────────
# Configuration de la page
# ─────────────────────────────────────────────

st.set_page_config(page_title="Assistant Escalade IA", page_icon="🧗", layout="wide")
st.title("🧗 Assistant IA Multi-Modèles — Données d'escalade")

# ─────────────────────────────────────────────
# Filtres et Configuration de la Sidebar (Nouveau Front)
# ─────────────────────────────────────────────

# Modification : Ajout de la plage de grade par défaut (0 à 80)
DEFAULT_FILTERS = {
    "country": None,
    "top_n": 10,
    "grade_range": (0, 80)
}

if "filters" not in st.session_state:
    st.session_state["filters"] = DEFAULT_FILTERS.copy()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

_f = st.session_state.get("filters", DEFAULT_FILTERS)

with st.sidebar:
    st.header("🤖 Configuration de l'IA")
    
    # Ajout de Gemma 4 dans le sélecteur graphique
    selected_model = st.selectbox(
        "Modèle de langage (LLM)",
        options=["gemma4:e4b", "llama3.2"],
        index=0,
        help="Sélectionnez le modèle local Ollama à interroger."
    )
    
    st.divider()
    st.header("🎛️ Filtres Données")

    top_n = st.slider(
        "Nombre de résultats à afficher",
        3, 30,
        value=_f.get("top_n", DEFAULT_FILTERS["top_n"]),
    )

    country = st.text_input(
        "Pays (code, ex: SWE, and, arg...)",
        value=_f.get("country") or "",
    )

    # NOUVEAU : Slider à double curseur pour sélectionner une plage de niveaux
    grade_range = st.slider(
        "Plage de niveau (Score numérique)",
        0, 80,
        value=_f.get("grade_range", DEFAULT_FILTERS["grade_range"]),
        help="Filtre les voies et les grimpeurs selon ce score (ex: 45 correspond environ à un 6b)."
    )

    # Mise à jour du session state avec le nouveau filtre
    st.session_state["filters"] = {
        "country": country.strip().lower() or None,
        "top_n": top_n,
        "grade_range": grade_range
    }

    st.divider()
    if st.button("🗑️ Effacer l'historique du chat"):
        st.session_state["messages"] = []
        st.rerun()

# ─────────────────────────────────────────────
# Initialisation dynamique du LLM choisi
# ─────────────────────────────────────────────

llm = ChatOpenAI(
    model=selected_model,
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    temperature=0,
)

# ─────────────────────────────────────────────
# Agent : connexion aux serveurs MCP
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """Tu es un assistant expert en données d'escalade.
Tu as accès à trois sources de données via des outils MCP :
1. Grimpeurs : profils, performances, progression par pays
2. Grades : table de conversion entre grades numériques et notation française
3. Voies : caractéristiques des voies d'escalade par pays, crag et cluster

Réponds toujours en français. Utilise les outils pour répondre précisément aux questions.
Quand tu cites des grades numériques, propose leur équivalent en notation française si utile.
"""

async def run_agent(question: str):
    climber_script = MCP_DIR / "mcp_climber.py"
    grades_script = MCP_DIR / "mcp_grades.py"
    routes_script = MCP_DIR / "mcp_routes.py"

    for script in (climber_script, grades_script, routes_script):
        if not script.exists():
            raise FileNotFoundError(f"Serveur MCP introuvable : {script}")

    client = MultiServerMCPClient({
        "climber": {"command": "python", "args": [os.path.abspath(str(climber_script))], "transport": "stdio"},
        "grades": {"command": "python", "args": [os.path.abspath(str(grades_script))], "transport": "stdio"},
        "routes": {"command": "python", "args": [os.path.abspath(str(routes_script))], "transport": "stdio"},
    })

    try:
        tools = await client.get_tools()
    except* Exception as eg:
        messages = "; ".join(str(e) for e in eg.exceptions)
        raise RuntimeError(f"Échec de connexion aux serveurs MCP : {messages}") from eg

    agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)
    result = await agent.ainvoke({"messages": [("user", question)]})

    answer = result["messages"][-1].content
    steps = result["messages"][:-1]
    return answer, steps

# ─────────────────────────────────────────────
# Extraction de filtres depuis la question
# ─────────────────────────────────────────────

def extraire_filtres(question: str) -> dict:
    prompt = (
        "Extrait les filtres de cette question sur des données d'escalade. "
        "Reponds UNIQUEMENT avec un JSON valide : "
        '{"country": null,  "top_n": null, "reset": false} '
        "Les valeurs et top_n doivent etre des nombres entiers ou null. "
        "country doit etre un code pays (ex: SWE, and, arg) ou null. "
        "Question : " + question
    )
    agent = create_react_agent(llm, [])
    resultat = agent.invoke({"messages": [("user", prompt)]})
    raw = resultat["messages"][-1].content

    try:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        return json.loads(m.group()) if m else {}
    except json.JSONDecodeError:
        return {}

def appliquer_filtres(nouveaux: dict):
    if nouveaux.get("reset"):
        st.session_state["filters"] = DEFAULT_FILTERS.copy()

    f = st.session_state.get("filters", DEFAULT_FILTERS.copy())

    def to_int(val):
        try:
            return int(float(str(val))) if val is not None else None
        except (ValueError, TypeError):
            return None

    if nouveaux.get("country"):
        f["country"] = nouveaux["country"].strip().lower()

    v = to_int(nouveaux.get("top_n"))
    if v is not None:
        f["top_n"] = v

    st.session_state["filters"] = f

# ─────────────────────────────────────────────
# Dashboard Graphique
# ─────────────────────────────────────────────

st.divider()
st.header("📊 Analyse du dataset")

country_filter = st.session_state["filters"].get("country")
# Récupération de la plage sélectionnée
g_min, g_max = st.session_state["filters"].get("grade_range", (0, 80))

routes_visu = routes_df.copy()
climbers_visu = climbers_df.copy()

# 1. Application du filtre Pays
if country_filter:
    routes_visu = routes_visu[routes_visu["country"].str.lower() == country_filter]
    climbers_visu = climbers_visu[climbers_visu["country"].str.lower() == country_filter]

# 2. NOUVEAU : Application du filtre Plage de Niveaux
routes_visu = routes_visu[(routes_visu["grade_mean"] >= g_min) & (routes_visu["grade_mean"] <= g_max)]
climbers_visu = climbers_visu[(climbers_visu["grades_mean"] >= g_min) & (climbers_visu["grades_mean"] <= g_max)]

col1, col2 = st.columns(2)

# Carte
with col1:
    st.subheader("🗺️ Répartition des falaises")
    map_df = (
        routes_visu[["crag", "country", "latitude", "longitude", "grade_mean", "grade_fra"]]
        .groupby(["crag", "country", "latitude", "longitude", "grade_fra"])
        .agg({"grade_mean": "mean"})
        .reset_index()
        .dropna()
    )

    if not map_df.empty:
        fig_map = px.scatter_mapbox(
            map_df,
            lat="latitude",
            lon="longitude",
            hover_name="crag",
            hover_data={"country": True, "grade_fra": True, "grade_mean": False},
            color="grade_mean",
            color_continuous_scale="Plasma",
            zoom=2,
            height=500,
            labels={"grade_mean": "Score Niveau", "grade_fra": "Grade"}
        )
        fig_map.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("Aucune falaise géolocalisée ne correspond à ces critères.")

# Répartition H/F
with col2:
    st.subheader("👥 Répartition H/F")
    sex_df = climbers_visu.copy()
    sex_df["sex_label"] = sex_df["sex"].map({0: "Hommes", 1: "Femmes"})

    if not sex_df.empty:
        fig_pie = px.pie(sex_df, names="sex_label")
        st.plotly_chart(fig_pie, use_container_width=True)

        stats = (
            sex_df.groupby("sex_label")
            .agg({"height": "median", "weight": "median", "grades_mean": "median"})
            .round(1)
        )
        stats.columns = ["Taille médiane", "Poids médian", "Niveau médian"]
        st.dataframe(stats, use_container_width=True)
    else:
        st.info("Aucun grimpeur ne correspond à ces critères.")

st.divider()

# Histogramme des grades
st.subheader("📈 Répartition des grades par pays")
if not routes_visu.empty:
    grade_country = (
        routes_visu
        .groupby(["country", "grade_fra", "grade_id_round"])
        .size()
        .reset_index(name="count")
        .sort_values("grade_id_round")
    )

    fig_grade = px.bar(
        grade_country,
        x="country",
        y="count",
        color="grade_fra",
        barmode="stack",
        labels={"country": "Pays", "count": "Nombre de voies", "grade_fra": "Grade"}
    )
    st.plotly_chart(fig_grade, use_container_width=True)
else:
    st.info("Aucune voie disponible pour générer l'histogramme.")

# Corrélation
st.subheader("🏋️ Corrélation taille / poids / niveau")
scatter_df = climbers_visu.copy()
scatter_df = scatter_df.dropna(subset=["height", "weight", "grades_mean"])

if not scatter_df.empty:
    fig_scatter = px.scatter(
        scatter_df,
        x="height",
        y="weight",
        color="grades_mean",
        size="grades_mean",
        hover_data={"country": True, "age": True, "years_cl": True, "grade_fra": True, "grades_mean": False},
        labels={"height": "Taille (cm)", "weight": "Poids (kg)", "grade_fra": "Grade Moyen"}
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
else:
    st.info("Données insuffisantes pour afficher le graphique de corrélation.")

# ─────────────────────────────────────────────
# Interface de chat
# ─────────────────────────────────────────────

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Posez votre question sur les données d'escalade...")

if question:
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.spinner("Extraction des filtres..."):
        appliquer_filtres(extraire_filtres(question))

    with st.spinner(f"L'agent réfléchit avec {selected_model}..."):
        question_enrichie = f"{question} (filtres actifs : {st.session_state['filters']})"
        try:
            answer, steps = asyncio.run(run_agent(question_enrichie))
        except Exception as e:
            st.error(f"Erreur lors de l'appel à l'agent ({selected_model}) : {e}")
            st.stop()

    st.session_state["messages"].append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.markdown(answer)
        with st.expander("🔍 Étapes de raisonnement de l'agent"):
            for step in steps:
                st.write(step)

    st.caption(f"Filtres actifs : {st.session_state['filters']} | Modèle utilisé : {selected_model}")