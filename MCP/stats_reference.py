"""
Statistiques de référence des 3 datasets
Permet de vérifier "à la main" les réponses données par l'agent Ollama.

Usage :
    python stats_reference.py
"""

import json
from pathlib import Path
import pandas as pd

# ─────────────────────────────────────────────
# Chemins (Data/ est un dossier voisin de MCP/)
# ─────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent / "Data"

CLIMBER_CSV = DATA_DIR / "climber_df.csv"
GRADES_CSV = DATA_DIR / "grades_conversion_table.csv"
ROUTES_CSV = DATA_DIR / "route_geo.csv"  # ou "routes_rated.csv" selon ton fichier final

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)


def section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_nulls(df: pd.DataFrame):
    """Affiche les colonnes contenant des valeurs nulles, ou un message si aucune."""
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    if nulls.empty:
        print("Valeurs nulles par colonne   : (aucune)")
    else:
        print(f"Valeurs nulles par colonne   :\n{nulls}")


# ─────────────────────────────────────────────
# Grimpeurs
# ─────────────────────────────────────────────

def stats_climbers():
    section("CLIMBER_DF.CSV — Grimpeurs")

    if not CLIMBER_CSV.exists():
        print(f"  ⚠ Fichier introuvable : {CLIMBER_CSV}")
        return

    df = pd.read_csv(CLIMBER_CSV)

    print(f"Nombre total de lignes       : {len(df)}")
    print(f"Colonnes                     : {list(df.columns)}")
    print_nulls(df)

    print("\n--- Statistiques numériques ---")
    print(df.describe().T[["count", "mean", "std", "min", "max"]].round(2))

    print("\n--- Répartition par pays (top 10) ---")
    print(df["country"].value_counts().head(10))

    print("\n--- Répartition par sexe ---")
    print(df["sex"].value_counts())

    print("\n--- Top 5 grimpeurs par grades_max ---")
    print(df.nlargest(5, "grades_max")[["user_id", "country", "sex", "age", "grades_max", "grades_mean"]])

    print("\n--- Progression (grades_last - grades_first) ---")
    prog = df["grades_last"] - df["grades_first"]
    print(f"  Moyenne      : {prog.mean():.2f}")
    print(f"  Max          : {prog.max()}")
    print(f"  Min          : {prog.min()}")
    print(f"  En hausse    : {(prog > 0).sum()}")
    print(f"  En baisse    : {(prog < 0).sum()}")
    print(f"  Stable       : {(prog == 0).sum()}")

    return df


# ─────────────────────────────────────────────
# Grades
# ─────────────────────────────────────────────

def stats_grades():
    section("GRADES_CONVERSION_TABLE.CSV — Table de conversion")

    if not GRADES_CSV.exists():
        print(f"  ⚠ Fichier introuvable : {GRADES_CSV}")
        return

    df = pd.read_csv(GRADES_CSV, index_col=0)

    print(f"Nombre total de grades       : {len(df)}")
    print(f"Colonnes                     : {list(df.columns)}")
    print(f"grade_id min / max           : {df['grade_id'].min()} / {df['grade_id'].max()}")
    print_nulls(df)

    print("\n--- Aperçu (10 premiers / 10 derniers) ---")
    print(pd.concat([df.head(10), df.tail(10)]))

    return df


# ─────────────────────────────────────────────
# Voies
# ─────────────────────────────────────────────

def stats_routes():
    section("ROUTE_GEO.CSV — Voies d'escalade")

    if not ROUTES_CSV.exists():
        print(f"  ⚠ Fichier introuvable : {ROUTES_CSV}")
        return

    df = pd.read_csv(ROUTES_CSV, index_col=0)

    print(f"Nombre total de voies        : {len(df)}")
    print(f"Colonnes                     : {list(df.columns)}")
    print_nulls(df)

    print("\n--- Statistiques numériques ---")
    numeric_cols = df.select_dtypes(include="number").columns
    print(df[numeric_cols].describe().T[["count", "mean", "std", "min", "max"]].round(3))

    print("\n--- Répartition par pays (top 10) ---")
    print(df["country"].value_counts().head(10))

    print(f"\nNombre de crags uniques      : {df['crag'].nunique()}")
    print(f"Nombre de pays uniques       : {df['country'].nunique()}")

    if "cluster" in df.columns:
        print("\n--- Répartition par cluster ---")
        print(df["cluster"].value_counts().sort_index())

    if "rating_tot" in df.columns:
        print("\n--- Top 5 voies par rating ---")
        print(df.nlargest(5, "rating_tot")[["country", "crag", "name", "grade_mean", "rating_tot"]])

    if "latitude" in df.columns and "longitude" in df.columns:
        n_geo = df["latitude"].notna().sum()
        print(f"\nVoies géocodées (lat/lon)    : {n_geo}/{len(df)} ({100*n_geo/len(df):.1f}%)")

    return df


# ─────────────────────────────────────────────
# Mode interactif : poser une question spécifique
# ─────────────────────────────────────────────

def quick_check():
    """
    Petit espace pour vérifier rapidement une valeur précise,
    par ex. après une réponse du LLM à valider.
    """
    section("VÉRIFICATION RAPIDE (exemples)")

    df_climbers = pd.read_csv(CLIMBER_CSV) if CLIMBER_CSV.exists() else None
    df_routes = pd.read_csv(ROUTES_CSV, index_col=0) if ROUTES_CSV.exists() else None

    if df_climbers is not None:
        # Exemple : combien de grimpeurs suédois ?
        n_swe = (df_climbers["country"] == "SWE").sum()
        print(f"Exemple - grimpeurs SWE      : {n_swe}")

        # Exemple : grimpeur avec user_id=38
        row = df_climbers[df_climbers["user_id"] == 38]
        if not row.empty:
            print(f"Exemple - user_id=38         :\n{row.to_string(index=False)}")

    if df_routes is not None:
        # Exemple : voies en Andorre
        n_and = (df_routes["country"] == "and").sum()
        print(f"\nExemple - voies en Andorre   : {n_and}")


# ─────────────────────────────────────────────
# Export JSON (optionnel, pratique pour comparer par script)
# ─────────────────────────────────────────────

def export_summary_json(output_path: str = "stats_summary.json"):
    """Exporte un résumé condensé en JSON, utile pour comparer automatiquement aux réponses du LLM."""
    summary = {}

    if CLIMBER_CSV.exists():
        df = pd.read_csv(CLIMBER_CSV)
        summary["climbers"] = {
            "total": int(len(df)),
            "age_mean": round(df["age"].mean(), 2),
            "grades_max_mean": round(df["grades_max"].mean(), 2),
            "countries": df["country"].value_counts().to_dict(),
            "top5_by_grade_max": df.nlargest(5, "grades_max")[["user_id", "grades_max"]].to_dict(orient="records"),
        }

    if GRADES_CSV.exists():
        df = pd.read_csv(GRADES_CSV, index_col=0)
        summary["grades"] = {
            "total": int(len(df)),
            "grade_id_min": int(df["grade_id"].min()),
            "grade_id_max": int(df["grade_id"].max()),
        }

    if ROUTES_CSV.exists():
        df = pd.read_csv(ROUTES_CSV, index_col=0)
        summary["routes"] = {
            "total": int(len(df)),
            "grade_mean_global": round(df["grade_mean"].mean(), 2),
            "countries": df["country"].value_counts().to_dict(),
            "crags_unique": int(df["crag"].nunique()),
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Résumé exporté dans : {output_path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("   STATISTIQUES DE RÉFÉRENCE — Vérification des réponses LLM")
    print("=" * 70)

    stats_climbers()
    stats_grades()
    stats_routes()
    quick_check()
    export_summary_json()

    print("\n" + "=" * 70)
    print("Terminé. Compare ces chiffres avec les réponses de l'agent.")
    print("=" * 70)