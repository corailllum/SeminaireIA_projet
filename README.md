# Séminaire IA – Projet Escalade

**Étudiante :** Charlotte Chanudet
**Correcteur :** Mehdi Ammi

---

# Présentation du projet

Ce projet a été réalisé dans le cadre du cours de **Séminaire IA**.

L'objectif était de concevoir un système utilisant l'intelligence artificielle sur une thématique libre. Le sujet choisi est l'escalade sportive.

Le projet consiste à développer une application permettant d'explorer et d'interroger un ensemble de données liées à l'escalade grâce à :

- un **tableau de bord interactif** présentant plusieurs visualisations ;
- un **chatbot IA** capable de répondre à des questions en langage naturel ;
- une architecture basée sur le protocole **MCP** permettant à l'agent IA d'accéder à différentes sources de données spécialisées.

L'objectif final est de proposer un assistant capable d'aider un grimpeur à mieux comprendre les données disponibles et de lui recommander des voies adaptées à son profil et à son niveau.

---

# Objectifs

Les principaux objectifs du projet sont :

- Mettre en place une architecture MCP complète.
- Utiliser un modèle de langage local via Ollama.
- Permettre l'interrogation de plusieurs jeux de données spécialisés.
- Fournir des visualisations interactives des données.
- Répondre à des questions complexes formulées en langage naturel.
- Proposer un système de recommandation de voies d'escalade adapté au niveau du grimpeur.

---

# Données utilisées

Les données proviennent d'un dataset public disponible sur Kaggle.
lien des données : https://www.kaggle.com/datasets/jordizar/climb-dataset?select=climber_df.csv

Le dataset contient des informations sur :

- les grimpeurs ;
- les niveaux de difficulté ;
- les voies d'escalade ;

## Fichiers utilisés

### climber_df.csv

Contient les informations relatives aux grimpeurs. Description de chaque colonne (traduite et adaptée depuis la page Kaggle du dataset) :

| Colonne | Description |
|---------|--------------|
| `user_id` | Identifiant unique du grimpeur |
| `country` | Pays du grimpeur |
| `sex` | Genre du grimpeur (0 = Homme, 1 = Femme) |
| `height` | Taille du grimpeur (cm) |
| `weight` | Poids du grimpeur (kg) |
| `age` | Âge du grimpeur |
| `years_cl` | Nombre d'années de pratique de l'escalade |
| `date_first` | Date de la première ascension enregistrée |
| `date_last` | Date de la dernière ascension enregistrée |
| `grades_count` | Nombre total de voies réalisées par le grimpeur |
| `grades_first` | Grade de la première ascension |
| `grades_last` | Grade de la dernière ascension |
| `grades_max` | Grade maximum atteint par le grimpeur |
| `grades_mean` | Grade moyen du grimpeur sur l'ensemble de ses ascensions |
| `year_first` | Année de la première ascension |
| `year_last` | Année de la dernière ascension |

### grades_conversion_table.csv

Table de correspondance entre :

| Colonne | Description |
|---------|--------------|
| `grade_id` | Identifiant numérique du grade |
| `grade_fra` | Grade équivalent en notation française (ex: 6a, 7b+) |

Cette table donne la conversion entre la notation numérique et la notation française des grades. Elle permet à l'agent IA de convertir automatiquement les grades techniques dans un format compréhensible par les grimpeurs. Elle est désormais aussi directement fusionnée dans `mcp_routes.py` (voir plus bas) afin que les voies renvoient déjà leur grade en notation française, sans appel supplémentaire.

### routes_rated.csv

Contient les informations relatives aux voies. Description de chaque colonne (traduite et adaptée depuis la page Kaggle du dataset) :

| Colonne | Description |
|---------|--------------|
| `name_id` | Identifiant de la voie. Les données brutes d'ascensions ont été nettoyées et normalisées en amont (sur Kaggle) afin d'éviter qu'une même voie ou une même falaise n'apparaisse sous plusieurs noms différents. |
| `country` | Pays où se trouve la voie |
| `crag` | Falaise / site d'escalade |
| `sector` | Secteur au sein de la falaise |
| `name` | Nom de la voie |
| `grade_mean` | Grade moyen de la voie. Le calcul prend en compte les nuances de cotation données par les grimpeurs (un "7a dur" est compté comme un 7a/+, de même pour les cotations "molles"), puis la **médiane** de l'ensemble des cotations de la voie est utilisée plutôt que la moyenne, pour être plus robuste face aux valeurs aberrantes. |
| `rating_tot` | Note globale de la voie. Calculée à partir de trois variables (le sentiment exprimé dans les commentaires, la note donnée par les grimpeurs, et le taux de recommandation), réduites à une seule dimension via une analyse en composantes principales (PCA) — la note retenue est la première composante. |
| `tall_recommend_sum` | Indicateur de l'effet de la taille du grimpeur sur la difficulté perçue de la voie. Calculé en additionnant, pour chaque ascension : +1 si le grimpeur est grand (> 180 cm) et juge la voie facile ; -1 s'il est grand et la juge dure ; -1 si le grimpeur est petit (< 170 cm) et juge la voie facile ; +1 s'il est petit et la juge dure. Une valeur positive suggère une voie plus avantageuse pour les grimpeurs petits, une valeur négative pour les grimpeurs grands. |
| `cluster` | Catégorie de la voie parmi 9 clusters (0 à 8), obtenus par classification automatique des voies. Les profils identifiés sont : `0` voies faciles/douces ; `1` voies pour une raison ou une autre préférées par les femmes ; `2` voies célèbres ; `3` voies très difficiles ; `4` voies très répétées ; `5` voies "chippées" (prises artificiellement modifiées) à cotation douce ; `6` voies traditionnelles, non chippées ; `7` voies faciles à enchaîner à vue (on-sight), peu répétées ; `8` voies très célèbres mais peu répétées et peu traditionnelles. |

>  Le dataset original utilise le nom `rating_total` sur Kaggle ; la colonne a été renommée `rating_tot` dans les fichiers du projet, c'est ce dernier nom qui est utilisé dans le code.

### route_geo.csv

Ce fichier a été créé dans le cadre du projet.

À partir de `routes_rated.csv`, un script de géocodage a permis d'ajouter :

- la latitude ;
- la longitude.

Ces informations permettent de représenter les voies sur une carte interactive. **C'est désormais ce fichier (et non `routes_rated.csv`) qui est utilisé comme source de données par `mcp_routes.py`**, afin que les outils de recherche géographique (voies/falaises les plus proches) puissent fonctionner directement.

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

Chaque serveur MCP expose des outils pour interroger un ou plusieurs CSV. Tous les paramètres numériques des outils acceptent désormais indifféremment des entiers, des chaînes de caractères ou `null` (voir section *Difficultés rencontrées*), afin de fiabiliser les appels effectués par les LLM locaux.

### mcp_climber.py — Grimpeurs

| Outil | Description | Entrée | Sortie |
|-------|-------------|--------|--------|
| `get_climber_stats` | Statistiques globales | *(aucune)* | JSON : total grimpeurs, âge moyen, grade max moyen, grade moyen, répartition par sexe, nb de pays uniques, top 5 pays |
| `get_climbers_by_country` | Échantillon de grimpeurs filtrés par pays (limité à 20 résultats) | `country` (str, requis) | JSON : pays, nombre total, âge moyen, grade max moyen, échantillon de grimpeurs (max 20) |
| `get_top_climbers_by_country` | **(nouveau)** Top N grimpeurs d'un pays donné, triés par grade max — répond en un seul appel à des questions comme "les 5 meilleurs grimpeurs français" | `country` (str, requis), `n` (int, optionnel, défaut 10) | Liste JSON des N meilleurs grimpeurs du pays (user_id, country, sex, age, grades_max, grades_mean, years_cl) |
| `search_climbers` | **(nouveau)** Recherche multicritère combinable (taille, poids, genre, grade) — tous les critères sont optionnels, l'utilisateur peut n'en préciser qu'un seul | `min_height`, `max_height`, `min_weight`, `max_weight`, `sex`, `min_grade`, `max_grade`, `n` (tous optionnels) | JSON : critères appliqués, nombre total correspondant, nombre affiché, liste des grimpeurs (triée par grade décroissant) |
| `get_top_climbers` | Top N par grade maximum (toutes nationalités) | `n` (int, optionnel, défaut 10) | Liste JSON des N meilleurs grimpeurs (user_id, country, sex, age, grades_max, grades_mean, years_cl) |
| `get_climber_by_id` | Détail d'un grimpeur par user_id | `user_id` (int, requis) | JSON : toutes les colonnes du grimpeur correspondant |
| `get_progression_analysis` | Évolution grades_first → grades_last | *(aucune)* | JSON : progression moyenne/max/min, nombre de grimpeurs en progression/régression/stables |

### mcp_grades.py — Grades

| Outil | Description | Entrée | Sortie |
|-------|-------------|--------|--------|
| `get_all_grades` | Table complète de conversion | *(aucune)* | Liste JSON complète des couples grade_id ↔ grade_fra |
| `convert_grade_id_to_fra` | Numérique → notation française | `grade_id` (int, requis) | JSON : `{grade_id, grade_fra}` |
| `convert_grade_fra_to_id` | Notation française → numérique | `grade_fra` (str, requis) | JSON : `{grade_fra, grade_id}` |
| `get_grade_range` | Grades dans une plage d'ID | `min_id`, `max_id` (int, requis) | Liste JSON des grades dont l'ID est compris dans la plage |

### mcp_routes.py — Voies

| Outil | Description | Entrée | Sortie |
|-------|-------------|--------|--------|
| `get_route_stats` | Statistiques globales | *(aucune)* | JSON : total de voies, grade moyen/min/max, nb de pays uniques, top 5 pays, répartition par cluster, nb de crags uniques, rating moyen |
| `recommend_routes` | **(nouveau)** Recommande des voies adaptées à un niveau (débutant / intermédiaire / avancé, calculé par tertiles des grades), triées par note ; pays optionnel | `level` (str, optionnel, défaut `"debutant"`), `country` (str, optionnel), `n` (int, optionnel, défaut 5) | JSON : niveau retenu, seuils de grade calculés, nombre de voies correspondantes, liste des voies recommandées (avec grade_fra) |
| `get_routes_by_country` | Voies filtrées par pays (limité à 20 résultats) | `country` (str, requis) | JSON : pays, nombre de voies, grade moyen, liste des crags, échantillon de voies (max 20) |
| `get_top_rated_routes` | Top N par rating | `n` (int, optionnel, défaut 10) | Liste JSON des N voies les mieux notées (name_id, country, crag, name, grade_mean, cluster, rating_tot) |
| `get_routes_by_crag` | Voies d'un site spécifique | `crag` (str, requis) | Liste JSON de toutes les voies du site demandé |
| `get_routes_by_cluster` | Voies d'un cluster (0 à 8, voir signification des clusters dans la section *Données utilisées*) | `cluster` (int, requis) | JSON : cluster, nombre de voies, grade moyen, rating moyen, échantillon de voies (max 20) |
| `get_routes_by_grade_range` | Filtre par plage de grade | `min_grade`, `max_grade` (number, requis) | JSON : plage demandée, nombre de voies, échantillon de voies (max 20) |
| `get_nearest_routes` | **(nouveau)** N voies les plus proches d'un point GPS (latitude/longitude), distance calculée par haversine | `latitude`, `longitude` (number, requis), `n` (int, optionnel, défaut 10) | Liste JSON des N voies les plus proches, avec `distance_km` |
| `get_nearest_crags` | **(nouveau)** N falaises les plus proches d'un point GPS, voies regroupées par site | `latitude`, `longitude` (number, requis), `n` (int, optionnel, défaut 5) | Liste JSON des N falaises les plus proches (regroupées), avec `distance_km` et statistiques agrégées |

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
- gemma4:e4b

Le modèle principal utilisé pour l'application est **llama3.2**, exécuté localement via Ollama. `gemma4:e4b` a également été testé via un sélecteur de modèle dans l'interface, mais s'est montré plus sensible aux problèmes décrits ci-dessous (voir *Difficultés rencontrées*).

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
- Quels pays possèdent le plus de voies difficiles ?
- Quelle voie conseillerais-tu à un grimpeur débutant ?
- Quelles sont les voies les plus proches de telles coordonnées GPS ?

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

### Mauvaise gestion des paramètres

Au départ, chaque outil MCP ne couvrait qu'un seul type de filtre (par pays, par tri, etc.), ce qui obligeait l'agent à enchaîner plusieurs appels pour répondre à une question simple — par exemple "les 5 meilleurs grimpeurs français" nécessitait de filtrer par pays *puis* de trier par grade, en deux étapes de raisonnement distinctes. Cette multiplication des étapes augmentait le risque d'erreur et de perte de contexte, en particulier avec les modèles plus petits. La solution a été d'ajouter des outils combinés (`get_top_climbers_by_country`, `search_climbers`, `recommend_routes`) capables de filtrer, trier et limiter en un seul appel, avec des critères tous optionnels et combinables librement.

### Problème de typage

Les LLM locaux envoient parfois leurs paramètres dans un format inattendu : un nombre transmis comme chaîne de caractères (`"10"` au lieu de `10`), voire `null` explicite sur un paramètre pourtant requis. Ces écarts provoquaient des erreurs de validation côté serveur MCP (`Input validation error`) avant même que la logique applicative ne s'exécute. La correction a porté sur deux niveaux : des fonctions de conversion sûres (`to_int`, `to_float`, `to_str`) qui normalisent n'importe quelle entrée côté code, et des schémas JSON élargis (types multiples + `null` systématique) pour que la validation MCP elle-même n'empêche pas la requête d'atteindre ce code.

### Mise en place de Gemma (prompt pas assez précis)

Le passage à `gemma4:e4b` a révélé un comportement différent de llama3.2 : le modèle appelait bien les outils, mais finissait parfois par ignorer la question initiale et répondre par un message d'accueil générique, quelle que soit la question posée. L'analyse des étapes de raisonnement a montré que le modèle perdait le fil après avoir reçu un résultat d'outil trop volumineux, plutôt qu'un véritable refus d'utiliser les outils. Une partie de la solution relève du prompt système (à rendre plus directif sur la marche à suivre), et une autre partie relève directement de la taille des réponses des outils (voir point suivant).

### Bug à cause de la longueur de réponse JSON

Certains outils renvoyaient l'intégralité des lignes correspondant au filtre — par exemple tous les grimpeurs d'un pays, soit plusieurs centaines d'enregistrements JSON en une seule réponse. Cette volumétrie pouvait dépasser la fenêtre de contexte du modèle (`num_ctx` côté Ollama), provoquant une troncature silencieuse : le modèle perdait alors la question d'origine et produisait une réponse incohérente. La correction a consisté à plafonner systématiquement la taille des résultats retournés par les outils (`.head(20)` ou paramètre `n` explicite), afin de garder les réponses des outils légères et adaptées à des modèles à fenêtre de contexte réduite.

---

# Limites actuelles

Le projet présente encore certaines limites :

- Les recommandations sont basées sur des seuils statistiques (tertiles) plutôt que sur un véritable profil grimpeur ; le croisement avec les caractéristiques physiques d'un grimpeur reste à faire.
- Les réponses dépendent de la qualité des données disponibles.
- Certains modèles locaux (notamment les plus petits) peuvent encore produire des réponses imprécises sur des questions nécessitant plusieurs étapes de raisonnement.
- Les performances dépendent de la machine exécutant Ollama.

---

# Perspectives d'amélioration

Plusieurs évolutions sont envisagées :

- Recommandation de voies tenant compte du profil physique du grimpeur (taille, poids, morphologie).
- Recherche géographique avancée (déjà initiée avec `get_nearest_routes` / `get_nearest_crags`).
- Historique et suivi de progression des utilisateurs.
- Ajout de nouvelles sources de données.
- Évaluation automatique de la pertinence des réponses de l'agent.
- Déploiement sur un serveur accessible en ligne.
- Affiner le prompt système pour les modèles à fenêtre de contexte réduite (ex: gemma4:e4b).

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

Le résultat est un assistant spécialisé dans l'analyse de données d'escalade capable de répondre à des questions complexes en s'appuyant sur plusieurs sources de données structurées, avec une attention particulière portée à la robustesse des échanges entre l'agent et les serveurs MCP (typage, volumétrie des réponses, prise en charge de plusieurs modèles locaux).
