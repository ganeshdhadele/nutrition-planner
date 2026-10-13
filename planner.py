"""AI meal planner.

Approach (constraint-based heuristic optimisation + content-based recommendation):

  1. FILTER   - drop foods that violate the diet type, allergies or health-condition rules.
  2. GENERATE - for every meal, sample many candidate dish combinations from meal templates
                (e.g. roti + dal + sabzi + curd for lunch).
  3. OPTIMISE - for each candidate, search portion sizes (coordinate descent) to minimise a
                cost made of calorie/macro error, sodium, glycaemic index, fibre, and a
                repetition penalty that keeps the week varied.
  4. SELECT   - keep the lowest-cost candidate, then fine-tune the whole day against the
                daily targets.
  5. SWAPS    - find_alternatives() is a content-based recommender (nearest neighbours in
                standardised nutrient space) used to suggest similar foods.
Plans are reproducible: the same seed gives the same plan.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

from nutrition import Targets, UserProfile

_HERE = Path(__file__).parent
DATA_PATH = next((c for c in (_HERE / "data" / "foods.csv", _HERE / "foods.csv") if c.exists()),
                 _HERE / "data" / "foods.csv")
MEALS = ["Breakfast", "Lunch", "Snack", "Dinner"]
MEAL_SHARE = {"Breakfast": 0.25, "Lunch": 0.35, "Snack": 0.10, "Dinner": 0.30}
NUTRIENTS = ("kcal", "protein", "carbs", "fat", "fiber", "sodium")

FOOD_DIET_RANK = {"vegan": 0, "veg": 1, "egg": 2, "nonveg": 3}
USER_DIET_RANK = {"Vegan": 0, "Vegetarian": 1, "Eggetarian": 2, "Non-vegetarian": 3}

TEMPLATES = {
    "Breakfast": [["main"], ["main", "fruit"], ["main", "drink"],
                  ["main", "main"], ["main", "main", "drink"], ["main", "main", "fruit", "drink"]],
    "Lunch": [["carb", "protein", "veg", "side"], ["carb", "protein", "veg"],
              ["onepot", "side"], ["onepot", "protein", "side"],
              ["carb", "carb", "protein", "veg", "side"], ["onepot", "carb", "protein", "veg", "side"]],
    "Snack": [["snack"], ["snack", "fruit"], ["snack", "drink"], ["snack", "snack"],
              ["snack", "snack", "drink"], ["fruit", "drink"],
              ["snack", "snack", "fruit"], ["snack", "snack", "fruit", "drink"]],
    "Dinner": [["carb", "protein", "veg"], ["onepot", "veg"], ["soup", "carb", "protein"],
               ["carb", "protein", "soup"], ["carb", "carb", "protein", "veg"],
               ["onepot", "protein", "veg"], ["soup", "carb", "carb", "protein", "veg"]],
}
_S = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
ROLE_SCALES = {
    "main": _S, "onepot": _S, "protein": _S, "carb": _S,
    "snack": [0.5, 1.0, 1.5, 2.0], "veg": [0.5, 1.0, 1.5], "soup": [0.5, 1.0],
    "side": [0.5, 1.0], "drink": [0.5, 1.0], "fruit": [1.0, 2.0],
}
# Extra (larger) portion multipliers offered only to very high calorie targets, so that
# everyday plans keep natural portion sizes while bulking plans can still reach the target.
HIGH_KCAL_THRESHOLD = 2800
HIGH_KCAL_EXTRA = {"main": [2.5, 3.0], "onepot": [2.5, 3.0], "carb": [2.5, 3.0],
                   "snack": [2.5, 3.0], "protein": [2.5]}
ROLE_LABEL = {
    "main": "Breakfast items", "onepot": "One-pot meals", "carb": "Grains & breads",
    "protein": "Proteins & pulses", "veg": "Vegetables", "soup": "Soups",
    "side": "Dairy & sides", "drink": "Drinks", "snack": "Snacks", "fruit": "Fruits",
}


class PlannerError(Exception):
    """Raised when no valid plan can be built for the given profile."""


# --------------------------------------------------------------------------- data
def load_foods(path: Path | str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"name", "role", "meals", "diet", "serving", "kcal", "protein", "carbs",
                "fat", "fiber", "sodium", "gi", "allergens", "tags"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"foods file is missing columns: {sorted(missing)}")
    df["tags"] = df["tags"].fillna("").astype(str)
    df["meals"] = df["meals"].astype(str).str.split(";")
    df["allergens"] = df["allergens"].astype(str).apply(
        lambda s: set() if s == "none" else set(s.split(";")))
    df["tags"] = df["tags"].apply(lambda s: {t for t in s.split(";") if t})
    return df


def filter_foods(df: pd.DataFrame, profile: UserProfile) -> pd.DataFrame:
    """Remove foods that conflict with diet, allergies or health conditions."""
    allowed_rank = USER_DIET_RANK[profile.diet]
    mask = df["diet"].map(FOOD_DIET_RANK) <= allowed_rank
    banned = set(profile.allergies)
    mask &= df["allergens"].apply(lambda a: not (a & banned))
    if {"diabetes", "pcos"} & set(profile.conditions):
        mask &= df["gi"] != "high"
    if "hypertension" in profile.conditions:
        mask &= df["sodium"] <= 650
    return df[mask].reset_index(drop=True)


# ------------------------------------------------------------------ cost function
class _Ctx:
    """Weights and state shared across the search."""

    def __init__(self, profile: UserProfile, targets: Targets):
        conds = set(profile.conditions)
        self.targets = targets
        self.conds = conds
        self.sodium_w = 4.0 if "hypertension" in conds else 1.0
        self.gi_w = 1.5 if {"diabetes", "pcos"} & conds else 0.3
        self.fat_w = 2.0 if "cholesterol" in conds else 1.0
        self.carb_over_w = 2.0 if "diabetes" in conds else 1.0 if "pcos" in conds else 0.0
        self.big_portions = targets.kcal >= HIGH_KCAL_THRESHOLD
        self.last_used: dict[str, int] = {}
        self.today: set[str] = set()
        self.day = 0


def _allowed_scales(role: str, ctx: "_Ctx") -> list:
    return ROLE_SCALES[role] + (HIGH_KCAL_EXTRA.get(role, []) if ctx.big_portions else [])


def _totals(items, scales) -> dict:
    t = dict.fromkeys(NUTRIENTS, 0.0)
    for it, s in zip(items, scales):
        for k in NUTRIENTS:
            t[k] += it[k] * s
    return t


def _macro_cost(t: dict, tgt: dict, ctx: "_Ctx") -> float:
    over = max(0.0, t["carbs"] - tgt["carbs"]) / max(tgt["carbs"], 1)
    return (12.0 * abs(t["kcal"] - tgt["kcal"]) / tgt["kcal"]
            + 3.0 * abs(t["protein"] - tgt["protein"]) / max(tgt["protein"], 1)
            + 1.0 * abs(t["carbs"] - tgt["carbs"]) / max(tgt["carbs"], 1)
            + ctx.carb_over_w * over
            + ctx.fat_w * abs(t["fat"] - tgt["fat"]) / max(tgt["fat"], 1))


def _pref_cost(items, scales, ctx: _Ctx, share: float) -> float:
    """Preference penalties: sodium, glycaemic index, fibre, repetition, portions."""
    tg = ctx.targets
    t = _totals(items, scales)
    c = 0.0
    budget = tg.sodium_mg * share
    if t["sodium"] > budget:
        c += ctx.sodium_w * (t["sodium"] / budget - 1)
    c -= 0.3 * min(t["fiber"] / (tg.fiber_g * share), 1.0)
    for it, s in zip(items, scales):
        if it["gi"] == "medium":
            c += ctx.gi_w * 0.1
        elif it["gi"] == "high":
            c += ctx.gi_w * 0.4
        ago = ctx.day - ctx.last_used.get(it["name"], -99)
        c += 0.9 if ago == 1 else 0.5 if ago == 2 else 0.2 if ago <= 4 else 0.0
        if it["name"] in ctx.today:
            c += 1.5
        if "anemia" in ctx.conds and "iron" in it["tags"]:
            c -= 0.12
        c += 0.04 * abs(s - 1.0)  # prefer natural portion sizes
    return c


def _optimise(items, tgt: dict, ctx: _Ctx, share: float, passes: int = 3):
    """Coordinate descent over allowed portion multipliers."""
    scales = [1.0] * len(items)
    allowed = [_allowed_scales(it["role"], ctx) for it in items]

    def cost(sc):
        return _macro_cost(_totals(items, sc), tgt, ctx) + _pref_cost(items, sc, ctx, share)

    best = cost(scales)
    for _ in range(passes):
        improved = False
        for i in range(len(items)):
            for s in allowed[i]:
                if s == scales[i]:
                    continue
                trial = scales[:i] + [s] + scales[i + 1:]
                c = cost(trial)
                if c < best - 1e-9:
                    best, scales, improved = c, trial, True
        if not improved:
            break
    return scales, best


# ------------------------------------------------------------------- plan builder
def _candidates(pools: dict, meal: str, rng: random.Random, n: int = 80):
    templates = [t for t in TEMPLATES[meal] if all(pools.get(r) for r in t)]
    seen, out = set(), []
    for _ in range(n * 3):
        if len(out) >= n:
            break
        combo, used = [], set()
        for role in rng.choice(templates):
            options = [x for x in pools[role] if x["name"] not in used]
            if not options:
                break
            pick = rng.choice(options)
            used.add(pick["name"])
            combo.append(pick)
        else:
            key = tuple(sorted(used))
            if key not in seen:
                seen.add(key)
                out.append(combo)
    return out


def _polish_day(day_meals: dict, ctx: _Ctx, daily: dict, passes: int = 3) -> None:
    """Fine-tune portions across the whole day against the daily targets (in place)."""
    flat = [(m, i) for m, entries in day_meals.items() for i in range(len(entries))]
    tg = ctx.targets

    def cost() -> float:
        items = [day_meals[m][i][0] for m, i in flat]
        sc = [day_meals[m][i][1] for m, i in flat]
        c = 1.5 * _macro_cost(_totals(items, sc), daily, ctx)
        t = _totals(items, sc)
        if t["sodium"] > tg.sodium_mg:
            c += ctx.sodium_w * (t["sodium"] / tg.sodium_mg - 1)
        for meal, entries in day_meals.items():
            mk = sum(it["kcal"] * s for it, s in entries)
            c += 0.5 * abs(mk - MEAL_SHARE[meal] * tg.kcal) / tg.kcal
        c += 0.04 * sum(abs(s - 1.0) for _, s in (day_meals[m][i] for m, i in flat))
        return c

    best = cost()
    for _ in range(passes):
        improved = False
        for m, i in flat:
            item, cur = day_meals[m][i]
            for s in _allowed_scales(item["role"], ctx):
                if s == cur:
                    continue
                day_meals[m][i] = (item, s)
                c = cost()
                if c < best - 1e-9:
                    best, cur, improved = c, s, True
                else:
                    day_meals[m][i] = (item, cur)
            day_meals[m][i] = (item, cur)
        if not improved:
            break


def _scaled_item(it: dict, s: float) -> dict:
    return {
        "name": it["name"], "role": it["role"], "scale": s,
        "portion": f"{s:g} x {it['serving']}",
        "kcal": round(it["kcal"] * s), "protein": round(it["protein"] * s, 1),
        "carbs": round(it["carbs"] * s, 1), "fat": round(it["fat"] * s, 1),
        "fiber": round(it["fiber"] * s, 1), "sodium": round(it["sodium"] * s),
        "gi": it["gi"], "tags": sorted(it["tags"]), "allergens": sorted(it["allergens"]),
        "serving": it["serving"],
    }


def build_plan(profile: UserProfile, targets: Targets, foods: pd.DataFrame | None = None,
               days: int = 7, seed: int | None = None) -> list[dict]:
    """Build a multi-day meal plan. Returns a list of day dicts."""
    profile.validate()
    foods = load_foods() if foods is None else foods
    allowed = filter_foods(foods, profile)
    if allowed.empty:
        raise PlannerError("No foods are left after applying your diet, allergy and health filters.")
    records = allowed.to_dict("records")

    pools_by_meal = {}
    for meal in MEALS:
        pools = {}
        for r in records:
            if meal.lower() in r["meals"]:
                pools.setdefault(r["role"], []).append(r)
        if not any(all(pools.get(role) for role in t) for t in TEMPLATES[meal]):
            raise PlannerError(
                f"Not enough suitable foods to build {meal} with your diet / allergy / health "
                "settings. Try relaxing one of the restrictions.")
        pools_by_meal[meal] = pools

    rng = random.Random(seed)
    ctx = _Ctx(profile, targets)
    daily = {"kcal": targets.kcal, "protein": targets.protein_g,
             "carbs": targets.carbs_g, "fat": targets.fat_g}
    plan = []

    for d in range(days):
        ctx.day, ctx.today = d, set()
        day_meals: dict[str, list] = {}
        consumed = dict.fromkeys(daily, 0.0)
        for idx, meal in enumerate(MEALS):
            share = MEAL_SHARE[meal]
            remaining_share = sum(MEAL_SHARE[m] for m in MEALS[idx:])
            # Adaptive re-targeting: spread what is still missing from the day over the
            # meals that are left, so an early shortfall/overshoot gets corrected later.
            tgt = {k: max(0.3 * v * share, (v - consumed[k]) * share / remaining_share)
                   for k, v in daily.items()}
            best_combo, best_scales, best_cost = None, None, float("inf")
            for combo in _candidates(pools_by_meal[meal], meal, rng):
                scales, cost = _optimise(combo, tgt, ctx, share)
                if cost < best_cost:
                    best_combo, best_scales, best_cost = combo, scales, cost
            day_meals[meal] = list(zip(best_combo, best_scales))
            ctx.today.update(it["name"] for it in best_combo)
            got = _totals(best_combo, best_scales)
            for k in consumed:
                consumed[k] += got[k]

        _polish_day(day_meals, ctx, daily)

        meals_out, meal_totals = {}, {}
        for meal, entries in day_meals.items():
            meals_out[meal] = [_scaled_item(it, s) for it, s in entries]
            meal_totals[meal] = _totals([e[0] for e in entries], [e[1] for e in entries])
            for it, _ in entries:
                ctx.last_used[it["name"]] = d
        flat = [e for entries in day_meals.values() for e in entries]
        totals = _totals([e[0] for e in flat], [e[1] for e in flat])
        plan.append({"day": d + 1, "label": f"Day {d + 1}", "meals": meals_out,
                     "meal_totals": meal_totals, "totals": totals})
    return plan


# ----------------------------------------------------------------------- outputs
def plan_to_dataframe(plan: list[dict]) -> pd.DataFrame:
    rows = []
    for d in plan:
        for meal, items in d["meals"].items():
            for it in items:
                rows.append({"Day": d["label"], "Meal": meal, "Food": it["name"],
                             "Portion": it["portion"], "Calories": it["kcal"],
                             "Protein (g)": it["protein"], "Carbs (g)": it["carbs"],
                             "Fat (g)": it["fat"], "Fiber (g)": it["fiber"],
                             "Sodium (mg)": it["sodium"]})
    return pd.DataFrame(rows)


def grocery_list(plan: list[dict]) -> pd.DataFrame:
    """Aggregate every food in the plan into a shopping summary."""
    agg: dict[str, dict] = {}
    for d in plan:
        for items in d["meals"].values():
            for it in items:
                a = agg.setdefault(it["name"], {"Category": ROLE_LABEL[it["role"]],
                                                "Serving size": it["serving"],
                                                "Times used": 0, "Total servings": 0.0})
                a["Times used"] += 1
                a["Total servings"] += it["scale"]
    df = pd.DataFrame([{"Food": k, **v} for k, v in agg.items()])
    df["Total servings"] = df["Total servings"].round(2)
    return df.sort_values(["Category", "Food"]).reset_index(drop=True)


def find_alternatives(foods: pd.DataFrame, food_name: str, profile: UserProfile,
                      k: int = 5) -> pd.DataFrame:
    """Content-based recommender: nearest foods in standardised nutrient space,
    restricted to the same role / meal slot and the user's diet, allergies and conditions."""
    base_rows = foods[foods["name"] == food_name]
    if base_rows.empty:
        raise ValueError(f"Unknown food: {food_name}")
    base = base_rows.iloc[0]
    feats = ["kcal", "protein", "carbs", "fat", "fiber"]
    std = foods[feats].std().replace(0, 1)
    pool = filter_foods(foods, profile)
    pool = pool[(pool["role"] == base["role"]) & (pool["name"] != food_name)]
    pool = pool[pool["meals"].apply(lambda m: bool(set(m) & set(base["meals"])))]
    if pool.empty:
        return pd.DataFrame()
    diff = (pool[feats].to_numpy(dtype=float) - base[feats].to_numpy(dtype=float)) / std.to_numpy()
    sim = 1 / (1 + np.sqrt((diff ** 2).sum(axis=1)))
    out = pool.assign(Similarity=sim).sort_values("Similarity", ascending=False).head(k)

    def why(r) -> str:
        notes = []
        if r["fiber"] >= base["fiber"] + 1:
            notes.append("more fibre")
        if r["sodium"] <= base["sodium"] * 0.8:
            notes.append("less sodium")
        if r["protein"] >= base["protein"] + 3:
            notes.append("more protein")
        if r["kcal"] <= base["kcal"] * 0.85:
            notes.append("fewer calories")
        return ", ".join(notes) if notes else "similar nutrition"

    return pd.DataFrame({
        "Food": out["name"], "Serving": out["serving"], "Calories": out["kcal"],
        "Protein (g)": out["protein"], "Carbs (g)": out["carbs"], "Fat (g)": out["fat"],
        "Fiber (g)": out["fiber"], "Sodium (mg)": out["sodium"],
        "Similarity": out["Similarity"].round(2), "Why swap": out.apply(why, axis=1),
    }).reset_index(drop=True)
