# 🥗 AI Nutrition Planner & Health Recommendation System

A Python + Streamlit project that turns a person's body stats, goal, diet and health conditions into:

- a **personalised nutrition profile** (BMI, BMR, TDEE, calories, protein / carbs / fat / fibre / sodium / water targets),
- a **7-day Indian-friendly meal plan** (breakfast, lunch, snack, dinner) with portion sizes,
- **health recommendations** (lifestyle, condition-specific, diet-specific, and plan-driven),
- a **plan quality score**, **grocery list**, **healthier food swaps** and downloadable reports.

> ⚠️ Educational project, **not medical advice**. Adults (18+) only. Anyone with a medical condition, on medication, or pregnant should consult a doctor or registered dietitian.

---

## Quick start

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py              # opens http://localhost:8501
```

Run the tests (standard library `unittest`, no extra tools needed):

```bash
python -m unittest discover -s tests -v
```

Preview real output without running the app: see `sample_output/`.

---

## How it works (the "AI")

The system combines three well-understood techniques. There is **no trained neural network**; everything is transparent and explainable, which makes it easy to defend in a viva or report.

| Component | Technique | File |
|---|---|---|
| Nutrition engine | Clinical formulas: Mifflin-St Jeor BMR, activity multipliers (TDEE), 7700 kcal/kg, WHO Asia-Pacific BMI cut-offs, condition-aware macro rules | `nutrition.py` |
| Meal planner | **Constraint-based heuristic optimisation**: filter → generate candidate meals from templates → portion search (coordinate descent) minimising a cost function → adaptive re-targeting across the day → day-level fine-tuning | `planner.py` |
| Swap suggestions | **Content-based recommender**: nearest neighbours in standardised nutrient space, restricted to safe foods | `planner.py` |
| Health advice | **Rule-based expert system** driven by profile, conditions and plan analysis | `health.py` |

### Meal planner pipeline

```
 profile ──► targets (kcal, macros, fibre, sodium)
                │
 foods.csv ──► FILTER  (diet, allergies, diabetes/PCOS → no high-GI, hypertension → sodium cap)
                │
              for each day, for each meal:
                │   GENERATE ~80 candidate dish combinations from meal templates
                │            e.g. Lunch = roti + dal + sabzi + curd
                │   OPTIMISE portion multipliers (0.5x … 2x, up to 3x for very high targets)
                │            cost = calories + protein + carbs + fat error
                │                 + sodium + glycaemic index + fibre + repetition penalties
                │   RETARGET remaining meals using what is still missing from the day
                ▼
              POLISH whole day against daily targets  →  7-day plan
                │
                ▼
         analyse (score 0-100) → recommendations → grocery list → reports
```

### Health condition rules

| Condition | Effect on the plan |
|---|---|
| Diabetes / pre-diabetes | High-GI foods removed, carbs capped near 45% of calories, extra penalty for carb overshoot, fibre favoured |
| PCOS | High-GI foods removed, higher protein target |
| Hypertension | Sodium limit 1,500 mg (vs 2,300 mg), foods above 650 mg/serving removed, strong sodium penalty |
| High cholesterol | Fat capped at 25% of calories, stronger fat penalty |
| Anemia | Iron-rich foods favoured |

Allergies (gluten, dairy, nuts, egg, soy, fish, sesame) and diet type (vegan / vegetarian / eggetarian / non-vegetarian) are **hard filters**: a food that violates them is never selected.

### Plan quality score (0-100)

Calories 30 · Protein 20 · Carbs 7.5 · Fat 7.5 · Fibre 15 · Sodium 10 · Variety 10.

---

## Project structure

```
nutrition_planner/
├── app.py              Streamlit UI (6 tabs)
├── nutrition.py        BMI / BMR / TDEE / macro targets / weight projection
├── planner.py          Food filtering, meal optimiser, grocery list, swap recommender
├── health.py           Plan analysis + recommendation expert system
├── report.py           Markdown report builder
├── data/foods.csv      81 foods with nutrition, diet, allergens, GI, tags
├── tests/test_core.py  26 unit + randomised robustness tests
├── sample_output/      Example report and meal-plan CSV
└── requirements.txt
```

### Food dataset (`data/foods.csv`)

Columns: `name, role, meals, diet, serving, kcal, protein, carbs, fat, fiber, sodium, gi, allergens, tags`

- `role`: main, onepot, carb, protein, veg, soup, side, drink, snack, fruit (how a dish is used in a meal)
- `meals`: slots where it may appear (`breakfast;lunch;snack;dinner`)
- `diet`: vegan < veg < egg < nonveg
- `allergens`: `;`-separated or `none` · `tags`: iron, calcium, probiotic, omega3

Values are **approximate typical home-cooked portions** (compiled from common nutrition references; sodium includes usual cooking salt). To extend the system, just add rows to the CSV. A test checks that each food's calories agree with its macros.

---

## Tabs in the app

1. **Overview**: BMI, BMR, TDEE, targets, macro split, weight projection chart
2. **Meal plan**: pick a day; every meal with portions, calories and macros vs target
3. **Health tips**: prioritised recommendations (⚠️ high · 💡 medium · ✅ low)
4. **Plan quality**: score, breakdown, average vs target table, flags, calories per day
5. **Grocery & export**: weekly shopping summary; download meal plan CSV, grocery CSV, full Markdown report
6. **Food swaps**: similar, safe alternatives for any food in the plan

Use **🔀 Different plan** to generate another valid plan (plans are reproducible per seed).

---

## Test results (development)

- 26 tests: formulas against hand-computed values, filter rules, safety (no diet/allergy violations), calorie accuracy, reproducibility, recommender, report, plus randomised profiles.
- Stress test on 200 random profiles (all diets, allergies, conditions, goals): all plans built, **0 diet/allergy violations**, mean weekly calorie error **0.6%**, **98.9%** of individual days within ±10% of target.

## Known limitations

- Dataset is small (81 dishes) and India-centric; accuracy depends on its approximate values.
- Plans for people with several conditions have fewer food choices; carbohydrate share for diabetic profiles can stay above the 45% target (the plan flags this).
- Sodium values include typical cooking salt, so strict low-sodium targets are hard to meet; the app flags this and advises cooking with less salt.
- Weight projection is a simple energy-balance model.
- Not validated clinically.

## Ideas to extend

- Larger food database (e.g. IFCT / USDA FoodData Central) and micronutrients (vitamins, minerals)
- Learn user preferences (ratings) with collaborative filtering
- Conversational coach using the Claude API on top of the generated plan
- Progress tracking (weight log, adherence) and weekly plan adaptation
- PDF export and multi-language support

## Tech

Python 3.10+, Streamlit, pandas, numpy.
