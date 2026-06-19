# SeminaireIA_projet

# Projet Escalade - Architecture MCP + Ollama

## Structure du projet

```
projet_escalade/
│
├── données/
│   ├── climber_df.csv
│   ├── grade_conversion_table.csv
│   └── route.csv
│
├── mcp_climber.py          ← Serveur MCP : données grimpeurs
├── mcp_grades.py           ← Serveur MCP : table de conversion des grades
├── mcp_routes.py           ← Serveur MCP : données des voies
│
├── ollama_agent.py         ← Agent ReAct (llama3.2 via Ollama + LangGraph)
├── test_mcp_agent.py       ← Tests unitaires et d'intégration
│
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Prérequis : Ollama

1. Installer Ollama : https://ollama.com
2. Télécharger le modèle :
```bash
ollama pull llama3.2
```
3. Vérifier qu'Ollama tourne :
```bash
ollama serve
```

---

## Serveurs MCP

Chaque serveur MCP expose des outils pour interroger un CSV.

### mcp_climber.py — Grimpeurs
| Outil | Description |
|-------|-------------|
| `get_climber_stats` | Statistiques globales |
| `get_climbers_by_country` | Filtre par pays (SWE, NOR, GBR…) |
| `get_top_climbers` | Top N par grade maximum |
| `get_climber_by_id` | Détail d'un grimpeur par user_id |
| `get_progression_analysis` | Évolution grades_first → grades_last |

### mcp_grades.py — Grades
| Outil | Description |
|-------|-------------|
| `get_all_grades` | Table complète de conversion |
| `convert_grade_id_to_fra` | Numérique → notation française |
| `convert_grade_fra_to_id` | Notation française → numérique |
| `get_grade_range` | Grades dans une plage d'ID |

### mcp_routes.py — Voies
| Outil | Description |
|-------|-------------|
| `get_route_stats` | Statistiques globales |
| `get_routes_by_country` | Filtre par pays (and, arg, fra…) |
| `get_top_rated_routes` | Top N par rating |
| `get_routes_by_crag` | Voies d'un site spécifique |
| `get_routes_by_cluster` | Voies d'un cluster (0-3) |
| `get_routes_by_grade_range` | Filtre par plage de grade |

---

## Lancer l'agent

```python
from ollama_agent import build_agent, ask_agent

agent = build_agent()
reponse = ask_agent("Quels sont les 5 meilleurs grimpeurs ?", agent)
print(reponse)
```

Ou directement :
```bash
python ollama_agent.py
```

---

## Lancer les tests

```bash
python test_mcp_agent.py
```

Les tests vérifient :
- Chargement des CSV
- Logique de chaque outil MCP (filtres, stats, conversions)
- Cohérence inter-données (grades valides, colonnes présentes)
- Connexion Ollama (skippé si Ollama n'est pas lancé)

---

## Étape suivante

Interface Streamlit avec :
- Graphiques Plotly (distribution des grades, carte des pays, clusters…)
- Chat avec l'agent Ollama intégré
