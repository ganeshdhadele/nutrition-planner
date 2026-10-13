"""Health recommendation system.

Two parts:
  1. analyze_plan()            - scores how well a generated meal plan meets the targets.
  2. generate_recommendations() - rule-based expert system producing personalised
                                  lifestyle / nutrition advice from the profile, the
                                  conditions and the plan analysis.
Educational only - not a substitute for professional medical advice.
"""
from __future__ import annotations

from nutrition import Targets, UserProfile, CONDITIONS

KEYS = ("kcal", "protein", "carbs", "fat", "fiber", "sodium")


def _targets_dict(t: Targets) -> dict:
    return {"kcal": t.kcal, "protein": t.protein_g, "carbs": t.carbs_g,
            "fat": t.fat_g, "fiber": t.fiber_g, "sodium": t.sodium_mg}


def analyze_plan(plan: list[dict], targets: Targets, profile: UserProfile) -> dict:
    """Average the plan and score it 0-100 against the targets."""
    n = len(plan)
    avg = {k: sum(d["totals"][k] for d in plan) / n for k in KEYS}
    tgt = _targets_dict(targets)
    adherence = {k: avg[k] / tgt[k] * 100 for k in KEYS}
    unique_foods = len({it["name"] for d in plan for m in d["meals"].values() for it in m})

    def closeness(ratio: float, tol: float) -> float:
        return max(0.0, 1 - abs(1 - ratio) / tol)

    breakdown = {
        "Calories": 30 * closeness(avg["kcal"] / tgt["kcal"], 0.15),
        "Protein": 20 * closeness(avg["protein"] / tgt["protein"], 0.25),
        "Carbs": 7.5 * closeness(avg["carbs"] / tgt["carbs"], 0.30),
        "Fat": 7.5 * closeness(avg["fat"] / tgt["fat"], 0.30),
        "Fibre": 15 * min(1.0, avg["fiber"] / tgt["fiber"]),
        "Sodium": 10 * (1.0 if avg["sodium"] <= tgt["sodium"]
                        else max(0.0, 1 - (avg["sodium"] - tgt["sodium"]) / tgt["sodium"])),
        "Variety": 10 * min(1.0, unique_foods / 25),
    }
    score = round(sum(breakdown.values()))
    grade = ("Excellent" if score >= 90 else "Good" if score >= 75
             else "Fair" if score >= 60 else "Needs work")

    flags = []
    if adherence["kcal"] > 110 or adherence["kcal"] < 90:
        flags.append(("warning", f"Average calories are {adherence['kcal']:.0f}% of target. "
                                 "Try regenerating the plan or adjusting portions."))
    if avg["sodium"] > tgt["sodium"]:
        flags.append(("warning", f"Average sodium {avg['sodium']:.0f} mg exceeds your limit of "
                                 f"{tgt['sodium']} mg. Cook with less salt and skip pickles/papad."))
    if avg["fiber"] < tgt["fiber"]:
        flags.append(("info", f"Fibre averages {avg['fiber']:.0f} g vs target {tgt['fiber']} g. "
                              "Add salad, fruit or whole grains."))
    carb_pct = avg["carbs"] * 4 / avg["kcal"] * 100
    if "diabetes" in profile.conditions and carb_pct > 50:
        flags.append(("info", f"Carbohydrates average {carb_pct:.0f}% of calories (aim for about 45%). "
                              "Prefer millets, dal/legumes and vegetables, and pair carbs with protein."))
    if adherence["protein"] < 85:
        flags.append(("info", "Protein is below target - add dal, paneer/tofu, eggs, chicken or sprouts."))
    if not flags:
        flags.append(("success", "The plan meets your calorie, protein, fibre and sodium targets."))

    return {"avg": avg, "targets": tgt, "adherence": adherence, "score": score,
            "grade": grade, "breakdown": breakdown, "unique_foods": unique_foods, "flags": flags}


def steps_goal(profile: UserProfile) -> str:
    return {"Lose weight": "8,000-10,000 steps/day",
            "Maintain weight": "7,000-8,000 steps/day",
            "Gain weight": "6,000-8,000 steps/day"}[profile.goal]


def _rec(category: str, title: str, advice: str, priority: str = "medium") -> dict:
    return {"category": category, "title": title, "advice": advice, "priority": priority}


def generate_recommendations(profile: UserProfile, targets: Targets,
                             analysis: dict | None = None) -> list[dict]:
    """Rule-based personalised recommendations, sorted high -> low priority."""
    recs: list[dict] = []
    conds = set(profile.conditions)

    # --- Body composition / goal -------------------------------------------------
    if targets.bmi_category == "Underweight":
        recs.append(_rec("Weight", "BMI is in the underweight range",
                         "Eat calorie-dense, nutritious foods (nuts, paneer, milk, ghee in moderation) and "
                         "include strength training. Consider a check-up to rule out underlying causes.", "high"))
    elif targets.bmi >= 25:
        recs.append(_rec("Weight", "Gradual weight loss will lower health risk",
                         f"Aim for 0.25-0.5 kg/week. A healthy range for your height is "
                         f"{targets.ideal_low_kg}-{targets.ideal_high_kg} kg. Even a 5% loss improves "
                         "blood sugar, BP and cholesterol.", "high"))
    elif targets.bmi >= 23:
        recs.append(_rec("Weight", "You are in the 'at risk' BMI band",
                         "Small habit changes (more walking, fewer sugary drinks) help prevent further gain.", "medium"))

    # --- Universal lifestyle ----------------------------------------------------
    recs.append(_rec("Hydration", f"Drink about {targets.water_ml / 1000:.1f} litres of fluid daily",
                     "Spread it across the day; start with a glass of water on waking. Buttermilk, coconut "
                     "water and unsweetened tea count; sugary drinks do not.", "medium"))
    recs.append(_rec("Activity", f"Target {steps_goal(profile)} and 150 min of exercise per week",
                     "Mix brisk walking with 2-3 strength-training sessions per week to protect muscle "
                     "while changing weight." if profile.goal != "Maintain weight" else
                     "Mix brisk walking with 2 strength-training sessions per week.", "medium"))
    recs.append(_rec("Sleep", "Sleep 7-9 hours",
                     "Poor sleep raises hunger hormones and cravings. Keep a consistent schedule and finish "
                     "dinner 2-3 hours before bed.", "medium"))
    recs.append(_rec("Eating habits", "Spread protein across all meals",
                     "Include a protein source (dal, curd, paneer, eggs, tofu, chicken, fish) at every meal "
                     "for fullness and muscle maintenance.", "low"))
    recs.append(_rec("Eating habits", "Eat slowly and fill half your plate with vegetables",
                     "Mindful eating and vegetable-first plates reduce overeating without counting every calorie.", "low"))

    if profile.goal == "Lose weight":
        recs.append(_rec("Goal: weight loss", "Avoid crash diets",
                         "Your calorie target is a moderate deficit. Going far below it can cost muscle, "
                         "energy and nutrients. Re-weigh weekly and adjust every 2-4 weeks.", "medium"))
    elif profile.goal == "Gain weight":
        recs.append(_rec("Goal: weight gain", "Gain with a modest surplus",
                         "Add a snack of nuts, milk or banana-peanut smoothie, and lift weights so the gain "
                         "is mostly muscle rather than fat.", "medium"))

    # --- Conditions --------------------------------------------------------------
    if "diabetes" in conds:
        recs.append(_rec(CONDITIONS["diabetes"], "Choose low-glycaemic, high-fibre carbohydrates",
                         "Prefer millets (jowar, ragi, bajra), whole pulses, brown rice and vegetables. Pair carbs "
                         "with protein/fibre, keep meal times regular, and limit sweets, white rice and fruit juice. "
                         "Monitor blood sugar and talk to your doctor before changing medication.", "high"))
    if "hypertension" in conds:
        recs.append(_rec(CONDITIONS["hypertension"], f"Keep sodium under {targets.sodium_mg} mg/day",
                         "Reduce added salt, pickles, papad, namkeen, instant noodles and processed foods. "
                         "Potassium-rich foods (banana, coconut water, leafy greens) help unless your doctor "
                         "has restricted potassium. Follow a DASH-style pattern.", "high"))
    if "cholesterol" in conds:
        recs.append(_rec(CONDITIONS["cholesterol"], "Cut saturated fat and add soluble fibre",
                         "Limit fried food, ghee/butter and bakery items. Add oats, legumes, nuts, flaxseed and "
                         "fatty fish (or flax/walnuts if vegetarian). Fat is capped at 25% of calories in your plan.", "high"))
    if "pcos" in conds:
        recs.append(_rec(CONDITIONS["pcos"], "Prioritise low-GI carbs, protein and strength training",
                         "Avoid refined flour and sugary drinks, eat protein at each meal and exercise regularly. "
                         "Even 5-10% weight loss can improve cycles and insulin sensitivity.", "high"))
    if "anemia" in conds:
        recs.append(_rec(CONDITIONS["anemia"], "Boost iron absorption",
                         "Eat iron-rich foods (dal, rajma, chole, spinach, ragi, sprouts) with vitamin C (lemon, "
                         "amla, guava, orange). Avoid tea/coffee with meals. Get ferritin and B12 tested and follow "
                         "your doctor's supplement advice.", "high"))

    # --- Diet type ------------------------------------------------------------
    if profile.diet in ("Vegan", "Vegetarian"):
        recs.append(_rec("Nutrients", "Watch vitamin B12 and omega-3",
                         "B12 is scarce in plant foods - fortified foods or a supplement are usually needed "
                         "(confirm with a blood test). Add flaxseed, chia or walnuts for omega-3.", "medium"))
    elif profile.diet == "Non-vegetarian":
        recs.append(_rec("Nutrients", "Eat fish twice a week",
                         "Fatty fish provides omega-3. Prefer grilled/curried over deep-fried chicken and fish.", "low"))
    if profile.age >= 50:
        recs.append(_rec("Nutrients", "Support bones with calcium and vitamin D",
                         "Include milk/curd/ragi/tofu and get morning sunlight. Ask your doctor about "
                         "vitamin D testing.", "medium"))

    # --- Plan-driven ------------------------------------------------------------
    if analysis:
        avg, tgt = analysis["avg"], analysis["targets"]
        if avg["sodium"] > tgt["sodium"]:
            pr = "high" if "hypertension" in conds else "medium"
            recs.append(_rec("Plan check", "Plan sodium is above your limit",
                             f"Average {avg['sodium']:.0f} mg vs limit {tgt['sodium']} mg (values include typical "
                             "cooking salt). Cook with about half the usual salt and flavour with spices, lemon and herbs.", pr))
        if avg["fiber"] < tgt["fiber"]:
            recs.append(_rec("Plan check", "Add more fibre",
                             f"Fibre averages {avg['fiber']:.0f} g vs {tgt['fiber']} g target. "
                             "Add a salad, a fruit and swap refined grains for millets/whole grains.", "medium"))

    order = {"high": 0, "medium": 1, "low": 2}
    recs.sort(key=lambda r: order[r["priority"]])
    return recs
