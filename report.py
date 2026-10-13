"""Builds a plain-text / Markdown report of a generated plan (no Streamlit dependency)."""
from __future__ import annotations

from nutrition import ALLERGENS, CONDITIONS

DISCLAIMER = ("This plan is for educational purposes and general wellness. It is not medical advice. "
              "Consult a doctor or registered dietitian before changing your diet, especially if you "
              "have a medical condition, take medication, are pregnant, or are under 18.")


def build_report(result: dict) -> str:
    p, t, plan, a, recs = (result["profile"], result["targets"], result["plan"],
                           result["analysis"], result["recs"])
    L: list[str] = []
    L.append(f"# Nutrition & Health Plan for {p.name}\n")
    L.append("## Profile")
    L.append(f"- Age / sex: {p.age} / {p.sex}")
    L.append(f"- Height / weight: {p.height_cm:g} cm / {p.weight_kg:g} kg")
    L.append(f"- Activity: {p.activity}")
    L.append(f"- Goal: {p.goal}" + (f" ({p.pace_kg_per_week:g} kg/week)" if p.goal != "Maintain weight" else ""))
    L.append(f"- Diet: {p.diet}")
    L.append("- Allergies: " + (", ".join(ALLERGENS[x] for x in p.allergies) or "none"))
    L.append("- Conditions: " + (", ".join(CONDITIONS[x] for x in p.conditions) or "none"))
    L.append("\n## Daily targets")
    L.append(f"- BMI {t.bmi} ({t.bmi_category}); healthy weight range {t.ideal_low_kg}-{t.ideal_high_kg} kg")
    L.append(f"- BMR {t.bmr:.0f} kcal, TDEE {t.tdee:.0f} kcal")
    L.append(f"- Calories {t.kcal:.0f} kcal | Protein {t.protein_g} g | Carbs {t.carbs_g} g | Fat {t.fat_g} g")
    L.append(f"- Fibre {t.fiber_g} g | Sodium limit {t.sodium_mg} mg | Fluids {t.water_ml / 1000:.1f} L")
    L.append("\n## 7-day meal plan")
    for d in plan:
        tot = d["totals"]
        L.append(f"\n### {d['label']}  ({tot['kcal']:.0f} kcal, P {tot['protein']:.0f} g, "
                 f"C {tot['carbs']:.0f} g, F {tot['fat']:.0f} g)")
        for meal, items in d["meals"].items():
            mk = d["meal_totals"][meal]["kcal"]
            L.append(f"- **{meal}** ({mk:.0f} kcal): " + "; ".join(f"{i['name']} ({i['portion']})" for i in items))
    L.append(f"\n## Plan quality: {a['score']}/100 ({a['grade']})")
    for k, v in a["avg"].items():
        L.append(f"- Average {k}: {v:.0f} (target {a['targets'][k]:.0f}, {a['adherence'][k]:.0f}%)")
    L.append("\n## Personalised health recommendations")
    for r in recs:
        L.append(f"- **{r['title']}** [{r['category']}]: {r['advice']}")
    L.append(f"\n---\n{DISCLAIMER}\n")
    return "\n".join(L)
