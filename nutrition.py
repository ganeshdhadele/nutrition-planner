"""Nutrition science engine.

Calculates BMI, BMR, TDEE, calorie target, macro / fibre / sodium / water targets
from a user profile, plus a simple weight-change projection.

References (standard, widely used formulas):
  * Mifflin-St Jeor equation for BMR
  * Activity multipliers for TDEE
  * 7700 kcal ~ 1 kg body fat
  * WHO Asia-Pacific BMI cut-offs (better suited to Indian populations)
This is educational software, not medical advice.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

ACTIVITY_LEVELS = {
    "Sedentary (little or no exercise)": 1.2,
    "Lightly active (1-3 days/week)": 1.375,
    "Moderately active (3-5 days/week)": 1.55,
    "Very active (6-7 days/week)": 1.725,
    "Extra active (hard training / physical job)": 1.9,
}
GOALS = ("Lose weight", "Maintain weight", "Gain weight")
DIETS = ("Vegan", "Vegetarian", "Eggetarian", "Non-vegetarian")
SEXES = ("Male", "Female")
ALLERGENS = {
    "gluten": "Gluten (wheat)",
    "dairy": "Dairy",
    "nuts": "Peanuts / tree nuts",
    "egg": "Egg",
    "soy": "Soy",
    "fish": "Fish",
    "sesame": "Sesame",
}
CONDITIONS = {
    "diabetes": "Diabetes / pre-diabetes",
    "hypertension": "Hypertension (high BP)",
    "cholesterol": "High cholesterol",
    "pcos": "PCOS",
    "anemia": "Anemia (low iron)",
}
KCAL_PER_KG = 7700


@dataclass
class UserProfile:
    name: str = "User"
    age: int = 30
    sex: str = "Male"
    height_cm: float = 170.0
    weight_kg: float = 70.0
    activity: str = "Lightly active (1-3 days/week)"
    goal: str = "Maintain weight"
    pace_kg_per_week: float = 0.5
    diet: str = "Vegetarian"
    allergies: list = field(default_factory=list)
    conditions: list = field(default_factory=list)

    def validate(self) -> "UserProfile":
        errs = []
        if not 18 <= self.age <= 90:
            errs.append("Age must be between 18 and 90 (this planner is for adults).")
        if not 120 <= self.height_cm <= 230:
            errs.append("Height must be between 120 and 230 cm.")
        if not 30 <= self.weight_kg <= 250:
            errs.append("Weight must be between 30 and 250 kg.")
        if self.sex not in SEXES:
            errs.append(f"Sex must be one of {SEXES}.")
        if self.activity not in ACTIVITY_LEVELS:
            errs.append("Unknown activity level.")
        if self.goal not in GOALS:
            errs.append(f"Goal must be one of {GOALS}.")
        if self.diet not in DIETS:
            errs.append(f"Diet must be one of {DIETS}.")
        if not 0.1 <= self.pace_kg_per_week <= 1.0:
            errs.append("Pace must be between 0.1 and 1.0 kg per week.")
        bad = [a for a in self.allergies if a not in ALLERGENS]
        if bad:
            errs.append(f"Unknown allergens: {bad}")
        bad = [c for c in self.conditions if c not in CONDITIONS]
        if bad:
            errs.append(f"Unknown conditions: {bad}")
        if errs:
            raise ValueError(" ".join(errs))
        return self


@dataclass
class Targets:
    bmi: float
    bmi_category: str
    ideal_low_kg: float
    ideal_high_kg: float
    bmr: float
    tdee: float
    kcal: float
    protein_g: float
    carbs_g: float
    fat_g: float
    fiber_g: float
    sodium_mg: int
    water_ml: int
    warnings: list = field(default_factory=list)

    @property
    def protein_pct(self) -> float:
        return 4 * self.protein_g / self.kcal * 100

    @property
    def carbs_pct(self) -> float:
        return 4 * self.carbs_g / self.kcal * 100

    @property
    def fat_pct(self) -> float:
        return 9 * self.fat_g / self.kcal * 100


def bmi(weight_kg: float, height_cm: float) -> float:
    return weight_kg / (height_cm / 100) ** 2


def bmi_category(value: float) -> str:
    """WHO Asia-Pacific cut-offs."""
    if value < 18.5:
        return "Underweight"
    if value < 23:
        return "Normal"
    if value < 25:
        return "Overweight (at risk)"
    if value < 30:
        return "Obese (Class I)"
    return "Obese (Class II)"


def ideal_weight_range(height_cm: float) -> tuple[float, float]:
    """Weight range corresponding to BMI 18.5 - 22.9."""
    h2 = (height_cm / 100) ** 2
    return 18.5 * h2, 22.9 * h2


def bmr(p: UserProfile) -> float:
    """Mifflin-St Jeor equation (kcal/day)."""
    base = 10 * p.weight_kg + 6.25 * p.height_cm - 5 * p.age
    return base + (5 if p.sex == "Male" else -161)


def tdee(p: UserProfile) -> float:
    return bmr(p) * ACTIVITY_LEVELS[p.activity]


def calorie_target(p: UserProfile) -> float:
    t = tdee(p)
    if p.goal == "Lose weight":
        delta = -min(p.pace_kg_per_week * KCAL_PER_KG / 7, 0.25 * t)
    elif p.goal == "Gain weight":
        delta = min(p.pace_kg_per_week * KCAL_PER_KG / 7, 0.20 * t)
    else:
        delta = 0.0
    floor = 1500 if p.sex == "Male" else 1200
    return max(floor, t + delta)


def macro_targets(p: UserProfile, kcal: float) -> tuple[float, float, float]:
    """Return (protein_g, carbs_g, fat_g) for the calorie target."""
    h2 = (p.height_cm / 100) ** 2
    ref_weight = min(p.weight_kg, 25 * h2)  # avoid inflated protein for high BMI
    factor = {"Lose weight": 1.4, "Maintain weight": 1.1, "Gain weight": 1.5}[p.goal]
    if "pcos" in p.conditions:
        factor = max(factor, 1.4)
    if p.age >= 65:
        factor = max(factor, 1.2)
    protein_g = min(factor * ref_weight, 0.35 * kcal / 4)

    fat_pct = 0.25 if "cholesterol" in p.conditions else 0.30
    fat_g = fat_pct * kcal / 9
    carbs_g = (kcal - 4 * protein_g - 9 * fat_g) / 4

    if "diabetes" in p.conditions and 4 * carbs_g / kcal > 0.45:
        carbs_g = 0.45 * kcal / 4
        fat_g = (kcal - 4 * protein_g - 4 * carbs_g) / 9
    if carbs_g < 100:  # keep a sensible carb floor for brain / activity
        carbs_g = 100.0
        fat_g = max(20.0, (kcal - 4 * protein_g - 4 * carbs_g) / 9)
    return round(protein_g), round(carbs_g), round(fat_g)


def compute_targets(p: UserProfile) -> Targets:
    p.validate()
    b = bmi(p.weight_kg, p.height_cm)
    lo, hi = ideal_weight_range(p.height_cm)
    kcal = round(calorie_target(p) / 10) * 10
    protein, carbs, fat = macro_targets(p, kcal)
    fiber = max(25, round(14 * kcal / 1000))
    sodium = 1500 if "hypertension" in p.conditions else 2300
    water = 35 * p.weight_kg + (500 if "Very" in p.activity or "Extra" in p.activity else 0)
    water = int(round(water / 50) * 50)

    warnings = []
    if p.goal == "Lose weight" and b < 18.5:
        warnings.append("Your BMI is in the underweight range - weight loss is not advisable. "
                        "Consider 'Maintain' or 'Gain' and speak to a doctor.")
    if p.goal == "Gain weight" and b >= 25:
        warnings.append("Your BMI is already in the overweight range - a weight-gain target may not be "
                        "appropriate. Muscle gain at maintenance calories may suit you better.")
    if p.goal == "Lose weight" and calorie_target(p) <= (1500 if p.sex == "Male" else 1200):
        warnings.append("Calories were held at a safe minimum; the requested pace may be too aggressive.")
    if p.age >= 65:
        warnings.append("For adults over 65, please review this plan with a doctor or dietitian.")
    if p.conditions:
        warnings.append("You selected health conditions: use this plan as a guide and confirm it with "
                        "your doctor or registered dietitian.")

    return Targets(
        bmi=round(b, 1), bmi_category=bmi_category(b),
        ideal_low_kg=round(lo, 1), ideal_high_kg=round(hi, 1),
        bmr=round(bmr(p)), tdee=round(tdee(p)), kcal=kcal,
        protein_g=protein, carbs_g=carbs, fat_g=fat, fiber_g=fiber,
        sodium_mg=sodium, water_ml=water, warnings=warnings,
    )


def project_weight(p: UserProfile, kcal_target: float, weeks: int = 12) -> list[tuple[int, float]]:
    """Week-by-week weight projection. TDEE is recomputed as weight changes,
    so the effect of a deficit/surplus naturally tapers off."""
    w = p.weight_kg
    out = [(0, round(w, 2))]
    for wk in range(1, weeks + 1):
        t = tdee(replace(p, weight_kg=w))
        w += (kcal_target - t) * 7 / KCAL_PER_KG
        out.append((wk, round(w, 2)))
    return out
