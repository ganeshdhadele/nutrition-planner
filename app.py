"""AI Nutrition Planner & Health Recommendation System - Streamlit app.

Run with:  streamlit run app.py
"""
import pandas as pd
import streamlit as st

from health import analyze_plan, generate_recommendations
from nutrition import (ACTIVITY_LEVELS, ALLERGENS, CONDITIONS, DIETS, GOALS, SEXES,
                       UserProfile, compute_targets, project_weight)
from planner import (MEAL_SHARE, PlannerError, build_plan, find_alternatives,
                     grocery_list, load_foods, plan_to_dataframe)
from report import DISCLAIMER, build_report

st.set_page_config(page_title="AI Nutrition Planner", page_icon="🥗", layout="wide",
                   initial_sidebar_state="expanded")


@st.cache_resource
def get_foods() -> pd.DataFrame:
    return load_foods()


def generate(profile: UserProfile, seed: int):
    """Compute targets + plan + analysis + recommendations. Returns None on error."""
    try:
        targets = compute_targets(profile)
        plan = build_plan(profile, targets, get_foods(), seed=seed)
    except (ValueError, PlannerError) as exc:
        st.error(str(exc))
        return None
    analysis = analyze_plan(plan, targets, profile)
    recs = generate_recommendations(profile, targets, analysis)
    return {"profile": profile, "targets": targets, "plan": plan,
            "analysis": analysis, "recs": recs, "seed": seed}


# ------------------------------------------------------------------ sidebar form
with st.sidebar:
    st.header("Your profile")
    with st.form("profile_form"):
        name = st.text_input("Name", "Guest")
        c1, c2 = st.columns(2)
        age = c1.number_input("Age", min_value=18, max_value=90, value=30, step=1)
        sex = c2.selectbox("Sex", SEXES)
        height = c1.number_input("Height (cm)", min_value=120.0, max_value=230.0, value=170.0, step=1.0)
        weight = c2.number_input("Weight (kg)", min_value=30.0, max_value=250.0, value=70.0, step=0.5)
        activity = st.selectbox("Activity level", list(ACTIVITY_LEVELS), index=1)
        goal = st.selectbox("Goal", GOALS, index=1)
        pace = st.select_slider("Pace in kg/week (lose / gain goals)", options=[0.25, 0.5, 0.75, 1.0], value=0.5)
        diet = st.selectbox("Diet preference", DIETS, index=1)
        allergies = st.multiselect("Allergies / foods to avoid", list(ALLERGENS), format_func=ALLERGENS.get)
        conditions = st.multiselect("Health conditions", list(CONDITIONS), format_func=CONDITIONS.get)
        submitted = st.form_submit_button("Generate my plan", type="primary")

if submitted:
    profile = UserProfile(name=name.strip() or "Guest", age=int(age), sex=sex, height_cm=float(height),
                          weight_kg=float(weight), activity=activity, goal=goal,
                          pace_kg_per_week=float(pace), diet=diet,
                          allergies=list(allergies), conditions=list(conditions))
    st.session_state["result"] = generate(profile, seed=1)

result = st.session_state.get("result")

# ------------------------------------------------------------------ landing page
st.title("🥗 AI Nutrition Planner & Health Recommendation System")

if result is None:
    st.markdown(
        """
Fill in your details in the sidebar and press **Generate my plan**.
(On a phone, the sidebar is the panel that slides in from the left; tap the **>** arrow at the top left if it is hidden.)

**What you get**
- Your BMI, BMR, daily calories and macro / fibre / sodium / water targets
- A **7-day Indian-friendly meal plan** matched to your diet, allergies and health conditions
- Personalised **health recommendations** and a plan quality score
- A grocery list, healthier food swaps and downloadable reports

**How the AI works**: a constraint-based optimiser filters foods, builds candidate meals, and searches
portion sizes to hit your calorie and macro targets while limiting sodium and repetition. A
content-based recommender suggests food swaps.
"""
    )
    st.caption(DISCLAIMER)
    st.stop()

profile, targets, plan = result["profile"], result["targets"], result["plan"]
analysis, recs = result["analysis"], result["recs"]

top_l, top_r = st.columns([4, 1])
top_l.subheader(f"Plan for {profile.name}")
if top_r.button("🔀 Different plan"):
    new_seed = result["seed"] + 1
    fresh = generate(profile, seed=new_seed)
    if fresh is not None:
        st.session_state["result"] = fresh
        st.rerun()

for w in targets.warnings:
    st.warning(w)

tab_overview, tab_plan, tab_tips, tab_quality, tab_export, tab_swap = st.tabs(
    ["📊 Overview", "🍽️ Meal plan", "💡 Health tips", "📈 Plan quality", "🛒 Grocery & export", "🔁 Food swaps"])

# ------------------------------------------------------------------ overview
with tab_overview:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("BMI", f"{targets.bmi}", targets.bmi_category, delta_color="off")
    m2.metric("BMR", f"{targets.bmr:.0f} kcal")
    m3.metric("Maintenance (TDEE)", f"{targets.tdee:.0f} kcal")
    m4.metric("Daily target", f"{targets.kcal:.0f} kcal")
    st.caption(f"Healthy weight range for your height (BMI 18.5-22.9): "
               f"{targets.ideal_low_kg}-{targets.ideal_high_kg} kg")

    n1, n2, n3, n4, n5 = st.columns(5)
    n1.metric("Protein", f"{targets.protein_g} g")
    n2.metric("Carbs", f"{targets.carbs_g} g")
    n3.metric("Fat", f"{targets.fat_g} g")
    n4.metric("Fibre", f"{targets.fiber_g} g")
    n5.metric("Water", f"{targets.water_ml / 1000:.1f} L")

    left, right = st.columns(2)
    with left:
        st.markdown("**Calories from each macronutrient (%)**")
        macro_df = pd.DataFrame({"Share of calories (%)": [round(targets.protein_pct), round(targets.carbs_pct),
                                                           round(targets.fat_pct)]},
                                index=["Protein", "Carbs", "Fat"])
        st.bar_chart(macro_df)
    with right:
        weeks = st.slider("Weight projection (weeks)", 4, 26, 12)
        proj = project_weight(profile, targets.kcal, weeks)
        st.markdown("**Projected weight (kg)**")
        st.line_chart(pd.DataFrame({"Weight (kg)": [w for _, w in proj]}, index=[wk for wk, _ in proj]))
        st.caption("Estimate based on 7,700 kcal per kg and recalculated as weight changes. Real results vary.")

# ------------------------------------------------------------------ meal plan
with tab_plan:
    day_label = st.radio("Choose a day", [d["label"] for d in plan], horizontal=True)
    day = next(d for d in plan if d["label"] == day_label)
    tot = day["totals"]
    c = st.columns(5)
    c[0].metric("Calories", f"{tot['kcal']:.0f}", f"{tot['kcal'] - targets.kcal:+.0f} vs target", delta_color="off")
    c[1].metric("Protein", f"{tot['protein']:.0f} g", f"{tot['protein'] - targets.protein_g:+.0f} g", delta_color="off")
    c[2].metric("Carbs", f"{tot['carbs']:.0f} g", f"{tot['carbs'] - targets.carbs_g:+.0f} g", delta_color="off")
    c[3].metric("Fat", f"{tot['fat']:.0f} g", f"{tot['fat'] - targets.fat_g:+.0f} g", delta_color="off")
    c[4].metric("Fibre", f"{tot['fiber']:.0f} g", f"{tot['fiber'] - targets.fiber_g:+.0f} g", delta_color="off")

    for meal in MEAL_SHARE:
        items = day["meals"][meal]
        mt = day["meal_totals"][meal]
        st.markdown(f"#### {meal} - {mt['kcal']:.0f} kcal")
        rows = [{"Food": i["name"], "Portion": i["portion"], "Calories": i["kcal"],
                 "Protein (g)": i["protein"], "Carbs (g)": i["carbs"], "Fat (g)": i["fat"]} for i in items]
        st.dataframe(pd.DataFrame(rows), hide_index=True)

# ------------------------------------------------------------------ health tips
with tab_tips:
    st.markdown("Recommendations are generated from your profile, health conditions and plan analysis.")
    for r in recs:
        text = f"**{r['title']}** · _{r['category']}_\n\n{r['advice']}"
        if r["priority"] == "high":
            st.warning(text, icon="⚠️")
        elif r["priority"] == "medium":
            st.info(text, icon="💡")
        else:
            st.success(text, icon="✅")
    st.caption(DISCLAIMER)

# ------------------------------------------------------------------ plan quality
with tab_quality:
    q1, q2 = st.columns([1, 2])
    q1.metric("Plan score", f"{analysis['score']} / 100", analysis["grade"], delta_color="off")
    with q2:
        for level, msg in analysis["flags"]:
            {"warning": st.warning, "info": st.info, "success": st.success}[level](msg)

    labels = {"kcal": "Calories (kcal)", "protein": "Protein (g)", "carbs": "Carbs (g)",
              "fat": "Fat (g)", "fiber": "Fibre (g)", "sodium": "Sodium (mg)"}
    table = pd.DataFrame({
        "Nutrient": [labels[k] for k in labels],
        "Average / day": [round(analysis["avg"][k]) for k in labels],
        "Target": [round(analysis["targets"][k]) for k in labels],
        "% of target": [round(analysis["adherence"][k]) for k in labels],
    })
    st.dataframe(table, hide_index=True)
    st.markdown("**Score breakdown (points)**")
    st.bar_chart(pd.DataFrame({"Points": {k: round(v, 1) for k, v in analysis["breakdown"].items()}}))
    st.markdown("**Calories per day**")
    st.bar_chart(pd.DataFrame({"Calories": [round(d["totals"]["kcal"]) for d in plan]},
                              index=[d["label"] for d in plan]))
    st.caption(f"Distinct foods used this week: {analysis['unique_foods']}. "
               "Sodium values include typical home-cooking salt; cooking with less salt lowers them.")

# ------------------------------------------------------------------ grocery & export
with tab_export:
    st.markdown("#### Weekly grocery summary")
    groceries = grocery_list(plan)
    st.dataframe(groceries, hide_index=True)
    plan_df = plan_to_dataframe(plan)
    d1, d2, d3 = st.columns(3)
    d1.download_button("⬇️ Meal plan (CSV)", plan_df.to_csv(index=False).encode("utf-8-sig"),
                       file_name="meal_plan.csv", mime="text/csv")
    d2.download_button("⬇️ Grocery list (CSV)", groceries.to_csv(index=False).encode("utf-8-sig"),
                       file_name="grocery_list.csv", mime="text/csv")
    d3.download_button("⬇️ Full report (Markdown)", build_report(result).encode("utf-8"),
                       file_name="nutrition_report.md", mime="text/markdown")

# ------------------------------------------------------------------ food swaps
with tab_swap:
    st.markdown("Pick a food from your plan to see similar alternatives that respect your diet, "
                "allergies and health conditions (content-based recommender).")
    names = sorted({i["name"] for d in plan for ms in d["meals"].values() for i in ms})
    pick = st.selectbox("Food to swap", names)
    k = st.slider("How many alternatives", 3, 8, 5)
    alts = find_alternatives(get_foods(), pick, profile, k=k)
    if alts.empty:
        st.info("No suitable alternatives found for this food with your current settings.")
    else:
        st.dataframe(alts, hide_index=True)

st.divider()
st.caption(DISCLAIMER)
