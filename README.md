# Séminaire IA – Projet Escalade

**Étudiante :** Charlotte Chanudet
**Correcteur :** À compléter

---

# Présentation du projet

Ce projet a été réalisé dans le cadre du cours de **Séminaire IA**.

L'objectif était de concevoir un système utilisant l'intelligence artificielle sur une thématique libre. Le sujet choisi est l'escalade sportive.

Le projet consiste à développer une application permettant d'explorer et d'interroger un ensemble de données liées à l'escalade grâce à :

- un **tableau de bord interactif** présentant plusieurs visualisations ;
- un **chatbot IA** capable de répondre à des questions en langage naturel ;
- une architecture basée sur le protocole **MCP** permettant à l'agent IA d'accéder à différentes sources de données spécialisées.

L'objectif final est de proposer un assistant capable d'aider un grimpeur à mieux comprendre les données disponibles et, à terme, de lui recommander des voies adaptées à son profil et à son niveau.

---

# Objectifs

Les principaux objectifs du projet sont :

- Mettre en place une architecture MCP complète.
- Utiliser un modèle de langage local via Ollama.
- Permettre l'interrogation de plusieurs jeux de données spécialisés.
- Fournir des visualisations interactives des données.
- Répondre à des questions complexes formulées en langage naturel.
- Préparer une base permettant de développer un système de recommandation de voies d'escalade.

---

# Données utilisées

Les données proviennent d'un dataset public disponible sur Kaggle.

Le dataset contient des informations sur :

- les grimpeurs ;
- les niveaux de difficulté ;
- les voies d'escalade ;


## Fichiers utilisés

### climber_df.csv

Contient les informations relatives aux grimpeurs :

- user_id
- country
- sex
- height
- weight
- age
- years_cl
- date_first
- date_last
- grades_count
- grades_first
- grades_last
- grades_max
- grades_mean
- year_first
- year_last


### grades_conversion_table.csv

Table de correspondance entre :

- grade_id
- grade_fra

Cette table permet à l'agent IA de convertir automatiquement les grades techniques dans un format compréhensible par les grimpeurs.

### routes_rated.csv

Contient les informations relatives aux voies :

- name_id
- country
- crag
- sector
- name
- tall_recommend_sum
- grade_mean
- cluster
- rating_tot

### route_geo.csv

Ce fichier a été créé dans le cadre du projet.

À partir de `routes_rated.csv`, un script de géocodage a permis d'ajouter :

- la latitude ;
- la longitude.

Ces informations permettent de représenter les voies sur une carte interactive.

---

# Architecture du projet

Le projet repose sur une architecture MCP composée de plusieurs serveurs spécialisés.

Chaque serveur est responsable d'un domaine de données particulier.

```text
Utilisateur
      │
      ▼
 Interface Streamlit
      │
      ▼
 Agent IA (LangGraph + Ollama)
      │
 ┌────┼────┐
 ▼    ▼    ▼
MCP  MCP  MCP
Climber Grades Routes
```

L'agent choisit dynamiquement quels outils utiliser afin de répondre à la question de l'utilisateur.

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

# Technologies utilisées

**Intelligence artificielle**
- Ollama
- LangChain
- LangGraph
- MCP (Model Context Protocol)

**Analyse de données**

- Pandas
- NumPy

**Interface utilisateur**
- Streamlit
- Plotly

---

# Modèles testés

Au cours du projet, plusieurs modèles ont été expérimentés :

- llama3.2
- gemma4

Le modèle principal utilisé pour l'application est **llama3.2**, exécuté localement via Ollama.

---

# Fonctionnalités

## Dashboard interactif

L'application propose plusieurs visualisations :

### Carte des falaises

Affichage géographique des sites d'escalade à partir des coordonnées GPS.

### Répartition des niveaux

Visualisation de la distribution des grades présents dans le dataset.

### Analyse des grimpeurs

Statistiques sur :

- le sexe ;
- la taille ;
- le poids ;
- le niveau moyen.

### Corrélation morphologie / performance

Étude de la relation entre les caractéristiques physiques des grimpeurs et leur niveau.

## Assistant conversationnel

L'utilisateur peut poser des questions telles que :

- Quels sont les meilleurs grimpeurs suédois ?
- Quelles sont les voies les mieux notées en Argentine ?
- À quoi correspond le grade numérique 53 ?
- Quels pays pcossèdent le plus de voies difficiles ?

L'agent sélectionne automatiquement les outils MCP nécessaires pour construire sa réponse.

---

# Structure du projet

```text
projet_escalade/
│
├── Data/
│   ├── climber_df.csv
│   ├── geocode_cache.json
│   ├── geocode_routes.py
│   ├── grades_conversion_table.csv
│   ├── route_geo.csv
│   └── routes_rated.csv
│
├── FrontEnd/
│   ├── app.py
│   └── app2model.py
├── MCP/
│   ├── mcp_climber.py
│   ├── mcp_grades.py
│   ├── mcp_routes.py
│   ├── ollama_agent.py
│   ├── stat_reference.py
│   └── test_mcp_agent.py
│
├── requirements.txt
└── README.md
```

---
# Difficultés rencontrées
# Limites actuelles

Le projet présente encore certaines limites :

- Les recommandations personnalisées ne sont pas encore implémentées.
- Les réponses dépendent de la qualité des données disponibles.
- Certains modèles locaux peuvent produire des réponses imprécises.
- Les performances dépendent de la machine exécutant Ollama.

---

# Perspectives d'amélioration

Plusieurs évolutions sont envisagées :

- Système de recommandation de voies adapté au profil du grimpeur.
- Recherche géographique avancée.
- Historique et suivi de progression des utilisateurs.
- Ajout de nouvelles sources de données.
- Évaluation automatique de la pertinence des réponses de l'agent.
- Déploiement sur un serveur accessible en ligne.

---
---

# Installation

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


---

## Lancer l'application

```bash
streamlit run app.py
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

# Conclusion

Ce projet a permis de mettre en pratique plusieurs concepts étudiés dans le cadre du cours :

- utilisation d'un modèle de langage local ;
- architecture MCP ;
- orchestration d'outils avec LangGraph ;
- visualisation de données ;
- développement d'une interface utilisateur interactive.

Le résultat est un assistant spécialisé dans l'analyse de données d'escalade capable de répondre à des questions complexes en s'appuyant sur plusieurs sources de données structurées.



TODO 

- ajouté les longitude latitude au MCP
- amélorié le MY prompte
- les MCPs avec les soucie de numbers et chain de charatere
