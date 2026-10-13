"""Unit tests for the nutrition engine, planner, health module and report.

Run with either:
    python -m unittest discover -s tests -v
    pytest tests
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

import nutrition as nut
import planner as pl
from health import analyze_plan, generate_recommendations
from report import build_report

FOODS = pl.load_foods()

PROFILES = {
    "veg_lose": nut.UserProfile(name="A", age=28, sex="Female", height_cm=160, weight_kg=68,
                                goal="Lose weight", diet="Vegetarian"),
    "nonveg_gain": nut.UserProfile(name="B", age=24, sex="Male", height_cm=175, weight_kg=60,
                                   goal="Gain weight", activity="Very active (6-7 days/week)",
                                   diet="Non-vegetarian"),
    "vegan_diabetic": nut.UserProfile(name="C", age=52, sex="Male", height_cm=170, weight_kg=82,
                                      goal="Lose weight", diet="Vegan", allergies=["nuts"],
                                      conditions=["diabetes", "hypertension"]),
    "egg_pcos_anemia": nut.UserProfile(name="D", age=30, sex="Female", height_cm=158, weight_kg=60,
                                       goal="Maintain weight", diet="Eggetarian",
                                       allergies=["dairy", "gluten"], conditions=["pcos", "anemia"]),
}
DIET_RANK = pl.FOOD_DIET_RANK


class TestNutritionEngine(unittest.TestCase):
    def test_bmi_and_asian_categories(self):
        self.assertAlmostEqual(nut.bmi(70, 175), 22.86, places=2)
        self.assertEqual(nut.bmi_category(18.4), "Underweight")
        self.assertEqual(nut.bmi_category(22.9), "Normal")
        self.assertEqual(nut.bmi_category(23.0), "Overweight (at risk)")
        self.assertEqual(nut.bmi_category(25.0), "Obese (Class I)")
        self.assertEqual(nut.bmi_category(30.0), "Obese (Class II)")

    def test_mifflin_st_jeor_known_values(self):
        male = nut.UserProfile(age=30, sex="Male", height_cm=175, weight_kg=70)
        female = nut.UserProfile(age=30, sex="Female", height_cm=175, weight_kg=70)
        self.assertAlmostEqual(nut.bmr(male), 1648.75)
        self.assertAlmostEqual(nut.bmr(female), 1482.75)

    def test_goal_ordering_and_safety_floor(self):
        base = dict(age=30, sex="Female", height_cm=160, weight_kg=60)
        lose = nut.calorie_target(nut.UserProfile(goal="Lose weight", **base))
        keep = nut.calorie_target(nut.UserProfile(goal="Maintain weight", **base))
        gain = nut.calorie_target(nut.UserProfile(goal="Gain weight", **base))
        self.assertLess(lose, keep)
        self.assertLess(keep, gain)
        aggressive = nut.UserProfile(goal="Lose weight", pace_kg_per_week=1.0,
                                     activity="Sedentary (little or no exercise)", **base)
        self.assertGreaterEqual(nut.calorie_target(aggressive), 1200)
        male_small = nut.UserProfile(age=70, sex="Male", height_cm=150, weight_kg=45,
                                     goal="Lose weight", pace_kg_per_week=1.0)
        self.assertGreaterEqual(nut.calorie_target(male_small), 1500)

    def test_macros_add_up_to_calories(self):
        for key, p in PROFILES.items():
            t = nut.compute_targets(p)
            energy = 4 * t.protein_g + 4 * t.carbs_g + 9 * t.fat_g
            self.assertAlmostEqual(energy / t.kcal, 1.0, delta=0.05, msg=key)

    def test_condition_rules(self):
        t_dia = nut.compute_targets(PROFILES["vegan_diabetic"])
        self.assertLessEqual(t_dia.carbs_pct, 46)
        self.assertEqual(t_dia.sodium_mg, 1500)
        self.assertEqual(nut.compute_targets(PROFILES["veg_lose"]).sodium_mg, 2300)

    def test_validation(self):
        for bad in (dict(age=15), dict(height_cm=50), dict(weight_kg=500), dict(sex="X"),
                    dict(goal="Fly"), dict(diet="Carnivore"), dict(allergies=["lava"]),
                    dict(conditions=["flu"]), dict(pace_kg_per_week=3)):
            with self.assertRaises(ValueError, msg=str(bad)):
                nut.compute_targets(nut.UserProfile(**bad))

    def test_warnings(self):
        t = nut.compute_targets(nut.UserProfile(age=25, sex="Male", height_cm=180, weight_kg=55,
                                                goal="Lose weight"))
        self.assertTrue(any("underweight" in w.lower() for w in t.warnings))

    def test_weight_projection_direction(self):
        p = PROFILES["veg_lose"]
        t = nut.compute_targets(p)
        proj = nut.project_weight(p, t.kcal, 12)
        self.assertEqual(len(proj), 13)
        self.assertLess(proj[-1][1], proj[0][1])
        g = PROFILES["nonveg_gain"]
        self.assertGreater(nut.project_weight(g, nut.compute_targets(g).kcal, 12)[-1][1], g.weight_kg)


class TestFoodData(unittest.TestCase):
    def test_dataset_integrity(self):
        self.assertGreaterEqual(len(FOODS), 60)
        self.assertEqual(FOODS["name"].nunique(), len(FOODS))
        for col in ("kcal", "protein", "carbs", "fat", "fiber", "sodium"):
            self.assertTrue((FOODS[col] >= 0).all(), col)
        self.assertTrue(FOODS["role"].isin(pl.ROLE_SCALES).all())
        self.assertTrue(FOODS["diet"].isin(DIET_RANK).all())
        self.assertTrue(FOODS["gi"].isin(["low", "medium", "high"]).all())
        self.assertTrue(FOODS["meals"].apply(lambda m: set(m) <= {"breakfast", "lunch", "snack", "dinner"}).all())

    def test_calories_consistent_with_macros(self):
        calc = 4 * FOODS["protein"] + 4 * FOODS["carbs"] + 9 * FOODS["fat"]
        dev = ((FOODS["kcal"] - calc) / calc).abs()
        self.assertLess(dev.max(), 0.20)

    def test_filters(self):
        vegan = pl.filter_foods(FOODS, nut.UserProfile(diet="Vegan"))
        self.assertTrue((vegan["diet"] == "vegan").all())
        veg = pl.filter_foods(FOODS, nut.UserProfile(diet="Vegetarian"))
        self.assertFalse(veg["diet"].isin(["egg", "nonveg"]).any())
        egg = pl.filter_foods(FOODS, nut.UserProfile(diet="Eggetarian"))
        self.assertFalse((egg["diet"] == "nonveg").any())
        allergic = pl.filter_foods(FOODS, nut.UserProfile(diet="Non-vegetarian", allergies=["dairy", "fish"]))
        self.assertFalse(allergic["allergens"].apply(lambda a: bool(a & {"dairy", "fish"})).any())
        diabetic = pl.filter_foods(FOODS, nut.UserProfile(conditions=["diabetes"]))
        self.assertFalse((diabetic["gi"] == "high").any())
        bp = pl.filter_foods(FOODS, nut.UserProfile(conditions=["hypertension"]))
        self.assertTrue((bp["sodium"] <= 650).all())


class TestPlanner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.built = {}
        for key, p in PROFILES.items():
            t = nut.compute_targets(p)
            cls.built[key] = (p, t, pl.build_plan(p, t, FOODS, seed=3))

    def test_structure(self):
        for key, (_, _, plan) in self.built.items():
            self.assertEqual(len(plan), 7, key)
            for day in plan:
                self.assertEqual(list(day["meals"]), pl.MEALS)
                for items in day["meals"].values():
                    self.assertGreaterEqual(len(items), 1)
                    names = [i["name"] for i in items]
                    self.assertEqual(len(names), len(set(names)), "duplicate food inside one meal")

    def test_diet_allergy_and_condition_rules_respected(self):
        for key, (p, _, plan) in self.built.items():
            banned = set(p.allergies)
            for day in plan:
                for items in day["meals"].values():
                    for it in items:
                        row = FOODS[FOODS["name"] == it["name"]].iloc[0]
                        self.assertLessEqual(DIET_RANK[row["diet"]], pl.USER_DIET_RANK[p.diet], (key, it["name"]))
                        self.assertFalse(set(it["allergens"]) & banned, (key, it["name"]))
                        if {"diabetes", "pcos"} & set(p.conditions):
                            self.assertNotEqual(it["gi"], "high", (key, it["name"]))

    def test_calories_close_to_target(self):
        for key, (_, t, plan) in self.built.items():
            avg = sum(d["totals"]["kcal"] for d in plan) / len(plan)
            self.assertAlmostEqual(avg / t.kcal, 1.0, delta=0.10, msg=key)
            for d in plan:
                self.assertAlmostEqual(d["totals"]["kcal"] / t.kcal, 1.0, delta=0.15, msg=(key, d["label"]))

    def test_protein_reasonably_met(self):
        for key, (_, t, plan) in self.built.items():
            avg = sum(d["totals"]["protein"] for d in plan) / len(plan)
            self.assertGreater(avg / t.protein_g, 0.80, key)

    def test_variety(self):
        for key, (_, _, plan) in self.built.items():
            used = {i["name"] for d in plan for ms in d["meals"].values() for i in ms}
            self.assertGreaterEqual(len(used), 20, key)

    def test_totals_match_items(self):
        _, _, plan = self.built["veg_lose"]
        for d in plan:
            items_kcal = sum(i["kcal"] for ms in d["meals"].values() for i in ms)
            self.assertAlmostEqual(items_kcal, d["totals"]["kcal"], delta=len(d["meals"]) * 6)

    def test_reproducible_and_shuffle(self):
        p = PROFILES["veg_lose"]
        t = nut.compute_targets(p)

        def sig(plan):
            return [[i["name"] for ms in d["meals"].values() for i in ms] for d in plan]
        self.assertEqual(sig(pl.build_plan(p, t, FOODS, seed=11)), sig(pl.build_plan(p, t, FOODS, seed=11)))
        self.assertNotEqual(sig(pl.build_plan(p, t, FOODS, seed=11)), sig(pl.build_plan(p, t, FOODS, seed=12)))

    def test_impossible_profile_raises_friendly_error(self):
        p, t = PROFILES["veg_lose"], nut.compute_targets(PROFILES["veg_lose"])
        only_snacks = FOODS[FOODS["role"] == "snack"]
        with self.assertRaises(pl.PlannerError):
            pl.build_plan(p, t, only_snacks, seed=1)

    def test_dataframe_and_grocery(self):
        _, _, plan = self.built["veg_lose"]
        df = pl.plan_to_dataframe(plan)
        self.assertEqual(set(df["Day"]), {f"Day {i}" for i in range(1, 8)})
        self.assertEqual(set(df["Meal"]), set(pl.MEALS))
        groc = pl.grocery_list(plan)
        total_scale = sum(i["scale"] for d in plan for ms in d["meals"].values() for i in ms)
        self.assertAlmostEqual(groc["Total servings"].sum(), total_scale, places=1)
        self.assertEqual(groc["Times used"].sum(), len(df))


class TestAlternatives(unittest.TestCase):
    def test_alternatives_are_similar_safe_and_ranked(self):
        p = nut.UserProfile(diet="Vegetarian", allergies=["soy"])
        alts = pl.find_alternatives(FOODS, "Dal Tadka", p, k=4)
        self.assertEqual(len(alts), 4)
        self.assertNotIn("Dal Tadka", set(alts["Food"]))
        roles = FOODS.set_index("name").loc[alts["Food"], "role"]
        self.assertTrue((roles == "protein").all())
        self.assertNotIn("Soya Chunk Curry", set(alts["Food"]))
        self.assertTrue(alts["Similarity"].is_monotonic_decreasing)

    def test_unknown_food(self):
        with self.assertRaises(ValueError):
            pl.find_alternatives(FOODS, "Moon Cheese", nut.UserProfile())


class TestHealthAndReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p = PROFILES["vegan_diabetic"]
        t = nut.compute_targets(p)
        plan = pl.build_plan(p, t, FOODS, seed=5)
        cls.result = {"profile": p, "targets": t, "plan": plan}
        cls.result["analysis"] = analyze_plan(plan, t, p)
        cls.result["recs"] = generate_recommendations(p, t, cls.result["analysis"])

    def test_score_bounds_and_breakdown(self):
        a = self.result["analysis"]
        self.assertTrue(0 <= a["score"] <= 100)
        self.assertAlmostEqual(sum(a["breakdown"].values()), a["score"], delta=1)
        self.assertTrue(a["flags"])

    def test_recommendations_are_personalised_and_sorted(self):
        recs = self.result["recs"]
        text = " ".join(r["title"] + r["advice"] for r in recs).lower()
        self.assertIn("glycaemic", text)       # diabetes
        self.assertIn("sodium", text)          # hypertension
        self.assertIn("b12", text)             # vegan
        order = {"high": 0, "medium": 1, "low": 2}
        ranks = [order[r["priority"]] for r in recs]
        self.assertEqual(ranks, sorted(ranks))
        healthy = nut.UserProfile(diet="Non-vegetarian")
        plain = " ".join(r["title"] for r in generate_recommendations(healthy, nut.compute_targets(healthy))).lower()
        self.assertNotIn("glycaemic", plain)

    def test_report(self):
        md = build_report(self.result)
        for needle in ("# Nutrition & Health Plan", "## Daily targets", "### Day 7", "not medical advice",
                       "Personalised health recommendations"):
            self.assertIn(needle, md)


class TestRobustness(unittest.TestCase):
    """Randomised profiles: plans must always build, stay safe and stay close to the calorie target."""

    def test_random_profiles(self):
        import random
        rng = random.Random(2026)
        days = off_target = 0
        for i in range(40):
            p = nut.UserProfile(
                age=rng.randint(18, 80), sex=rng.choice(nut.SEXES), height_cm=rng.randint(145, 195),
                weight_kg=rng.randint(40, 120), activity=rng.choice(list(nut.ACTIVITY_LEVELS)),
                goal=rng.choice(nut.GOALS), pace_kg_per_week=rng.choice([0.25, 0.5, 0.75, 1.0]),
                diet=rng.choice(nut.DIETS), allergies=rng.sample(list(nut.ALLERGENS), rng.randint(0, 3)),
                conditions=rng.sample(list(nut.CONDITIONS), rng.randint(0, 3)))
            t = nut.compute_targets(p)
            plan = pl.build_plan(p, t, FOODS, seed=i)
            a = analyze_plan(plan, t, p)
            generate_recommendations(p, t, a)
            self.assertTrue(0 <= a["score"] <= 100)
            for d in plan:
                days += 1
                off_target += abs(d["totals"]["kcal"] / t.kcal - 1) > 0.10
                for items in d["meals"].values():
                    for it in items:
                        row = FOODS[FOODS["name"] == it["name"]].iloc[0]
                        self.assertLessEqual(DIET_RANK[row["diet"]], pl.USER_DIET_RANK[p.diet])
                        self.assertFalse(set(it["allergens"]) & set(p.allergies))
        self.assertLess(off_target / days, 0.05, f"{off_target}/{days} days more than 10% off target")


if __name__ == "__main__":
    unittest.main(verbosity=2)
