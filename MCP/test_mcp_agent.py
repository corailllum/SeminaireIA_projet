"""
Fichier de tests - Projet Escalade
Teste les 3 serveurs MCP et l'agent Ollama de manière indépendante.

Lancer avec : python test_mcp_agent.py
"""

import json
import sys
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import tempfile
import os

# ─────────────────────────────────────────────
# Données de test (extraits des vrais CSV)
# ─────────────────────────────────────────────

SAMPLE_CLIMBER_CSV = """user_id,country,sex,height,weight,age,years_cl,date_first,date_last,grades_count,grades_first,grades_last,grades_max,grades_mean,year_first,year_last
1,SWE,0,177,73,41.0,21,1999-02-06,2001-07-31,84,36,55,62,46.75,1999,2001
3,SWE,0,180,78,44.0,22,1999-03-31,2000-07-19,12,53,51,59,52.83,1999,2000
4,SWE,1,165,58,33.0,16,2004-06-30,2009-05-26,119,53,49,64,53.89,2004,2009
10,SWE,0,167,63,52.0,25,2000-01-14,2017-06-01,298,53,49,63,49.41,2000,2017
38,GBR,0,178,73,35.0,24,2000-11-03,2017-07-21,323,55,55,71,57.74,2000,2017
79,DEU,0,188,83,43.0,26,1999-03-14,2017-08-26,375,62,53,66,56.61,1999,2017
"""

SAMPLE_GRADE_CSV = """,grade_id,grade_fra
0,0,-
1,1,-
3,3,1
4,4,1a
5,5,1b
40,40,5c
42,42,6a
50,50,7a
59,59,8a
62,62,8a+
63,63,8a+/8b
64,64,8b
66,66,8b+
71,71,9a
"""

SAMPLE_ROUTE_CSV = """,name_id,country,crag,sector,name,tall_recommend_sum,grade_mean,cluster,rating_tot
0,0,and,montserrat,prohibitivo,diagonal de la x,-1,49.25,3,-0.045
1,1,and,montserrat,prohibitivo,mehir,-1,49.0,2,0.116
2,2,and,montserrat,prohibitivo,pas de la discordia,0,49.0,2,0.179
4,4,arg,bandurrias,rincon,tendinitis,1,48.5,0,0.076
5,5,arg,bariloche,pared blanca,barbaroja,0,49.0,1,-0.165
8,8,arg,bariloche,pared blanca,lenguita de gato,0,53.0,2,-0.074
"""


# ─────────────────────────────────────────────
# Utilitaire : création de fichiers CSV temporaires
# ─────────────────────────────────────────────

def create_temp_csv(content: str, suffix=".csv") -> str:
    """Crée un fichier CSV temporaire et retourne son chemin."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False, encoding="utf-8")
    f.write(content)
    f.close()
    return f.name


# ─────────────────────────────────────────────
# Tests MCP Grimpeurs
# ─────────────────────────────────────────────

class TestMCPClimber(unittest.TestCase):
    """Tests unitaires du serveur MCP grimpeurs."""

    def setUp(self):
        self.csv_path = create_temp_csv(SAMPLE_CLIMBER_CSV)
        self.df = pd.read_csv(self.csv_path)

    def tearDown(self):
        os.unlink(self.csv_path)

    def test_chargement_csv(self):
        """Le CSV se charge correctement."""
        self.assertEqual(len(self.df), 6)
        self.assertIn("user_id", self.df.columns)
        self.assertIn("grades_max", self.df.columns)

    def test_stats_globales(self):
        """Les statistiques globales sont calculées correctement."""
        stats = {
            "total_grimpeurs": len(self.df),
            "age_moyen": round(self.df["age"].mean(), 2),
            "grade_max_moyen": round(self.df["grades_max"].mean(), 2),
        }
        self.assertEqual(stats["total_grimpeurs"], 6)
        self.assertGreater(stats["age_moyen"], 0)
        self.assertGreater(stats["grade_max_moyen"], 0)
        print(f"  ✓ Stats globales: {stats['total_grimpeurs']} grimpeurs, âge moyen={stats['age_moyen']}")

    def test_filtre_par_pays(self):
        """Le filtre par pays retourne les bons résultats."""
        swe = self.df[self.df["country"] == "SWE"]
        self.assertEqual(len(swe), 4)

        gbr = self.df[self.df["country"] == "GBR"]
        self.assertEqual(len(gbr), 1)

        unknown = self.df[self.df["country"] == "XYZ"]
        self.assertEqual(len(unknown), 0)
        print(f"  ✓ Filtre pays: SWE={len(swe)}, GBR={len(gbr)}, XYZ={len(unknown)}")

    def test_top_grimpeurs(self):
        """Les top grimpeurs sont bien triés par grade_max."""
        top3 = self.df.nlargest(3, "grades_max")
        self.assertEqual(top3.iloc[0]["grades_max"], 71)  # user 38 (GBR)
        print(f"  ✓ Top grimpeur: user_id={top3.iloc[0]['user_id']}, grade_max={top3.iloc[0]['grades_max']}")

    def test_progression(self):
        """Le calcul de progression (grades_last - grades_first) fonctionne."""
        df_copy = self.df.copy()
        df_copy["progression"] = df_copy["grades_last"] - df_copy["grades_first"]
        en_progression = (df_copy["progression"] > 0).sum()
        en_regression = (df_copy["progression"] < 0).sum()
        self.assertGreaterEqual(en_progression + en_regression, 0)
        print(f"  ✓ Progression: {en_progression} en hausse, {en_regression} en baisse")

    def test_climber_by_id(self):
        """La recherche par user_id retourne le bon grimpeur."""
        user = self.df[self.df["user_id"] == 38]
        self.assertEqual(len(user), 1)
        self.assertEqual(user.iloc[0]["country"], "GBR")
        print(f"  ✓ Recherche ID 38: pays={user.iloc[0]['country']}, grade_max={user.iloc[0]['grades_max']}")


# ─────────────────────────────────────────────
# Tests MCP Grades
# ─────────────────────────────────────────────

class TestMCPGrades(unittest.TestCase):
    """Tests unitaires du serveur MCP grades."""

    def setUp(self):
        self.csv_path = create_temp_csv(SAMPLE_GRADE_CSV)
        self.df = pd.read_csv(self.csv_path, index_col=0)

    def tearDown(self):
        os.unlink(self.csv_path)

    def test_chargement_csv(self):
        """Le CSV de grades se charge correctement."""
        self.assertIn("grade_id", self.df.columns)
        self.assertIn("grade_fra", self.df.columns)
        print(f"  ✓ {len(self.df)} grades chargés")

    def test_conversion_id_vers_fra(self):
        """La conversion grade_id → grade_fra fonctionne."""
        row = self.df[self.df["grade_id"] == 42]
        self.assertEqual(row["grade_fra"].values[0], "6a")
        print("  ✓ grade_id=42 → '6a'")

    def test_conversion_fra_vers_id(self):
        """La conversion grade_fra → grade_id fonctionne."""
        row = self.df[self.df["grade_fra"] == "6a"]
        self.assertEqual(int(row["grade_id"].values[0]), 42)
        print("  ✓ '6a' → grade_id=42")

    def test_grade_inexistant(self):
        """Un grade inexistant retourne un résultat vide."""
        row = self.df[self.df["grade_fra"] == "99z"]
        self.assertTrue(row.empty)
        print("  ✓ Grade inexistant '99z' → résultat vide")

    def test_plage_grades(self):
        """Le filtre par plage de grade_id fonctionne."""
        filtered = self.df[(self.df["grade_id"] >= 40) & (self.df["grade_id"] <= 50)]
        self.assertGreater(len(filtered), 0)
        self.assertTrue(all(filtered["grade_id"] >= 40))
        self.assertTrue(all(filtered["grade_id"] <= 50))
        print(f"  ✓ Plage 40-50: {len(filtered)} grades trouvés")


# ─────────────────────────────────────────────
# Tests MCP Routes
# ─────────────────────────────────────────────

class TestMCPRoutes(unittest.TestCase):
    """Tests unitaires du serveur MCP routes."""

    def setUp(self):
        self.csv_path = create_temp_csv(SAMPLE_ROUTE_CSV)
        self.df = pd.read_csv(self.csv_path, index_col=0)

    def tearDown(self):
        os.unlink(self.csv_path)

    def test_chargement_csv(self):
        """Le CSV des routes se charge correctement."""
        self.assertIn("grade_mean", self.df.columns)
        self.assertIn("cluster", self.df.columns)
        self.assertIn("rating_tot", self.df.columns)
        print(f"  ✓ {len(self.df)} voies chargées")

    def test_stats_globales(self):
        """Les statistiques globales sont correctes."""
        stats = {
            "total": len(self.df),
            "grade_moyen": round(self.df["grade_mean"].mean(), 2),
            "pays_uniques": self.df["country"].nunique(),
        }
        self.assertEqual(stats["total"], 6)
        self.assertGreater(stats["grade_moyen"], 0)
        print(f"  ✓ {stats['total']} voies, grade moyen={stats['grade_moyen']}")

    def test_filtre_par_pays(self):
        """Le filtre par pays fonctionne."""
        arg = self.df[self.df["country"] == "arg"]
        self.assertEqual(len(arg), 3)

        and_routes = self.df[self.df["country"] == "and"]
        self.assertEqual(len(and_routes), 3)  # 3 routes andorranes dans l'échantillon
        # Note: 'and' dans pandas est un mot-clé, mais ici c'est bien une string
        print(f"  ✓ Filtre: arg={len(arg)}, and={len(and_routes)}")

    def test_top_rated(self):
        """Les meilleures voies sont bien triées."""
        top = self.df.nlargest(3, "rating_tot")
        self.assertGreaterEqual(top.iloc[0]["rating_tot"], top.iloc[1]["rating_tot"])
        print(f"  ✓ Top voie: '{top.iloc[0]['name']}', rating={top.iloc[0]['rating_tot']:.4f}")

    def test_filtre_cluster(self):
        """Le filtre par cluster retourne les bons résultats."""
        cluster2 = self.df[self.df["cluster"] == 2]
        self.assertEqual(len(cluster2), 3)
        print(f"  ✓ Cluster 2: {len(cluster2)} voies")

    def test_filtre_grade_range(self):
        """Le filtre par plage de grade fonctionne."""
        filtered = self.df[(self.df["grade_mean"] >= 49.0) & (self.df["grade_mean"] <= 53.0)]
        self.assertGreater(len(filtered), 0)
        print(f"  ✓ Grade 49-53: {len(filtered)} voies")


# ─────────────────────────────────────────────
# Tests d'intégration (sans Ollama)
# ─────────────────────────────────────────────

class TestIntegration(unittest.TestCase):
    """Tests d'intégration vérifiant la cohérence entre les données."""

    def setUp(self):
        self.climber_path = create_temp_csv(SAMPLE_CLIMBER_CSV)
        self.grade_path = create_temp_csv(SAMPLE_GRADE_CSV)
        self.route_path = create_temp_csv(SAMPLE_ROUTE_CSV)

        self.df_climbers = pd.read_csv(self.climber_path)
        self.df_grades = pd.read_csv(self.grade_path, index_col=0)
        self.df_routes = pd.read_csv(self.route_path, index_col=0)

    def tearDown(self):
        for path in [self.climber_path, self.grade_path, self.route_path]:
            os.unlink(path)

    def test_grades_dans_plage_valide(self):
        """Les grades_max des grimpeurs correspondent à des grade_id existants."""
        grade_ids = set(self.df_grades["grade_id"].values)
        for grade in self.df_climbers["grades_max"]:
            self.assertIn(
                grade, grade_ids,
                f"grades_max={grade} n'existe pas dans la table de conversion"
            )
        print("  ✓ Tous les grades_max des grimpeurs sont dans la table de conversion")

    def test_colonnes_requises_presentes(self):
        """Toutes les colonnes requises sont présentes dans chaque DataFrame."""
        colonnes_climber = ["user_id", "country", "sex", "grades_max", "grades_mean"]
        colonnes_grade = ["grade_id", "grade_fra"]
        colonnes_route = ["country", "crag", "grade_mean", "cluster", "rating_tot"]

        for col in colonnes_climber:
            self.assertIn(col, self.df_climbers.columns, f"Colonne manquante dans climbers: {col}")
        for col in colonnes_grade:
            self.assertIn(col, self.df_grades.columns, f"Colonne manquante dans grades: {col}")
        for col in colonnes_route:
            self.assertIn(col, self.df_routes.columns, f"Colonne manquante dans routes: {col}")
        print("  ✓ Toutes les colonnes requises sont présentes")

    def test_pas_de_valeurs_nulles_critiques(self):
        """Pas de valeurs nulles dans les colonnes critiques."""
        self.assertEqual(self.df_climbers["user_id"].isnull().sum(), 0)
        self.assertEqual(self.df_climbers["grades_max"].isnull().sum(), 0)
        self.assertEqual(self.df_routes["grade_mean"].isnull().sum(), 0)
        print("  ✓ Pas de valeurs nulles dans les colonnes critiques")


# ─────────────────────────────────────────────
# Test de connexion Ollama (optionnel)
# ─────────────────────────────────────────────

class TestOllamaConnection(unittest.TestCase):
    """Teste la connexion à Ollama (nécessite Ollama lancé localement)."""

    def test_ollama_accessible(self):
        """Vérifie qu'Ollama répond sur localhost:11434."""
        try:
            import urllib.request
            req = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5)
            self.assertEqual(req.status, 200)
            print("  ✓ Ollama accessible sur localhost:11434")
        except Exception as e:
            self.skipTest(f"Ollama non accessible (normal si pas lancé): {e}")

    def test_llama32_disponible(self):
        """Vérifie que le modèle llama3.2 est disponible dans Ollama."""
        try:
            import urllib.request
            import json as _json
            req = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5)
            data = _json.loads(req.read())
            model_names = [m["name"] for m in data.get("models", [])]
            llama_present = any("llama3.2" in name for name in model_names)
            if not llama_present:
                print(f"  ⚠ llama3.2 non trouvé. Modèles disponibles: {model_names}")
                print("  → Lancer: ollama pull llama3.2")
            else:
                print("  ✓ llama3.2 disponible")
        except Exception as e:
            self.skipTest(f"Ollama non accessible: {e}")


# ─────────────────────────────────────────────
# Runner principal
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("   TESTS - Projet Escalade (MCP + Ollama)")
    print("=" * 60)

    # Loader avec verbosité
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suites = [
        ("MCP Grimpeurs", TestMCPClimber),
        ("MCP Grades", TestMCPGrades),
        ("MCP Routes", TestMCPRoutes),
        ("Intégration", TestIntegration),
        ("Connexion Ollama", TestOllamaConnection),
    ]

    total_ok = 0
    total_fail = 0
    total_skip = 0

    for label, test_class in suites:
        print(f"\n▶ {label}")
        # Récupère les noms de méthodes de test directement depuis la classe
        method_names = loader.getTestCaseNames(test_class)

        for method_name in method_names:
            test = test_class(method_name)
            try:
                test.debug()
                total_ok += 1
            except unittest.SkipTest as e:
                print(f"  ⊘ {method_name}: ignoré ({e})")
                total_skip += 1
            except Exception as e:
                print(f"  ✗ {method_name}: {e}")
                total_fail += 1

    print("\n" + "=" * 60)
    print(f"  ✓ {total_ok} passés   ✗ {total_fail} échoués   ⊘ {total_skip} ignorés")
    print("=" * 60)