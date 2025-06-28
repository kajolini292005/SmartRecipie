import streamlit as st
import pandas as pd
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import matplotlib.pyplot as plt

# --- Streamlit page config ---
st.set_page_config(
    page_title="Smart Leftovers 🍱",
    layout="centered",  # Looks like a phone
    initial_sidebar_state="collapsed"
)

# --- Custom CSS for mobile-style UI ---
st.markdown("""
    <style>
    .reportview-container {
        padding: 0 1rem;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 400px;
        margin: auto;
        background-color: #ffffff;
        border-radius: 15px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        padding: 10px 0;
        border-radius: 8px;
        background-color: #2f8cff;
        color: white;
        font-size: 16px;
    }
    .stTextInput>div>div>input {
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Ingredient cleaning ---
def clean_ingredient(ing):
    ing = ing.lower()
    ing = re.sub(r'[^a-z\s]', '', ing)
    ing = re.sub(r'\s+', ' ', ing)
    return ing.strip()

# --- Check if recipe is veg or not ---
def is_veg_recipe(ingredients, non_veg_keywords):
    return not any(any(word in ing for word in non_veg_keywords) for ing in ingredients)

# --- Smart Suggestion with explanation ---
def smart_suggest(user_ingredients, df, tfidf_matrix, vectorizer, top_n=5, is_veg=True, max_ings=15):
    user_input = ' '.join(clean_ingredient(i) for i in user_ingredients)
    user_vec = vectorizer.transform([user_input])
    cosine_similarities = cosine_similarity(user_vec, tfidf_matrix).flatten()

    df['score'] = cosine_similarities
    df_filtered = df[df['ingredients'].apply(len) <= max_ings]
    
    if is_veg:
        df_filtered = df_filtered[df_filtered['ingredients'].apply(lambda ings: is_veg_recipe(ings, non_veg_keywords))]
    
    top_recipes = df_filtered.sort_values(by="score", ascending=False).head(top_n)
    results = []

    user_cleaned = [clean_ingredient(i) for i in user_ingredients]
    
    for _, row in top_recipes.iterrows():
        recipe_ings = [clean_ingredient(i) for i in row['ingredients']]
        matched = [i for i in user_cleaned if i in recipe_ings]
        unmatched = [i for i in recipe_ings if i not in matched]

        results.append({
            'Name': f"{row['cuisine'].title()} Dish #{row['id']}",
            'Cuisine': row['cuisine'].title(),
            'Matched': matched,
            'Unmatched': unmatched,
            'Score': round(row['score'], 3)
        })

    return results

# --- Load Dataset ---
@st.cache_data
def load_data():
    with open("train.json", "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df['ingredient_text'] = df['ingredients'].apply(lambda x: ' '.join(clean_ingredient(i) for i in x))
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['ingredient_text'])
    return df, tfidf_matrix, vectorizer

# --- Non-veg keywords list ---
non_veg_keywords = ["chicken", "fish", "mutton", "beef", "egg", "bacon", "meat", "pork", "shrimp", "lamb"]

# --- Tabs ---
tab1, tab2 = st.tabs(["🍽️ Recipe Recommender", "📊 Dashboard"])

# === TAB 1: RECIPE RECOMMENDER ===
with tab1:
    st.title("Smart Leftovers 🍱")
    st.write("Get recipes based on your leftover ingredients!")

    user_input = st.text_input("Enter ingredients (comma-separated)", "milk, egg, flour")
    is_veg_choice = st.selectbox("Do you want only vegetarian recipes?", ["Yes", "No"])
    max_ings = st.slider("Max number of ingredients per recipe", 3, 20, 10)

    if st.button("Suggest Recipes"):
        df, tfidf_matrix, vectorizer = load_data()
        user_ingredients = [i.strip() for i in user_input.split(",") if i.strip()]
        if user_ingredients:
            results = smart_suggest(
                user_ingredients,
                df,
                tfidf_matrix,
                vectorizer,
                top_n=5,
                is_veg=(is_veg_choice == "Yes"),
                max_ings=max_ings
            )
            if results:
                for r in results:
                    st.markdown(f"""
                        <div style='background-color:#f7f9fc; padding:10px 15px; border-radius:10px; margin-bottom:15px;'>
                            <b>{r['Name']} ({r['Cuisine']})</b><br>
                            ✅ Matched: {', '.join(r['Matched'])}<br>
                            ➕ Others: {', '.join(r['Unmatched'])}<br>
                            📈 Score: {r['Score']}
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No recipes matched your filters.")
        else:
            st.error("Please enter at least one ingredient.")

# === TAB 2: ANALYTICS DASHBOARD ===
with tab2:
    st.title("📊 Dataset Insights")

    df, _, _ = load_data()

    # 1. Most common ingredients
    all_ings = [clean_ingredient(i) for sublist in df['ingredients'] for i in sublist]
    common_ings = pd.Series(all_ings).value_counts().head(10)
    st.subheader("🔝 Top 10 Most Common Ingredients")
    st.bar_chart(common_ings)

    # 2. Top cuisines
    top_cuisines = df['cuisine'].value_counts().head(10)
    st.subheader("🌍 Top 10 Cuisines")
    st.bar_chart(top_cuisines)

    # 3. Average ingredient count
    st.subheader("📏 Average Ingredients per Recipe")
    avg_ings = df['ingredients'].apply(len).mean()
    st.metric("Average Ingredients", f"{avg_ings:.2f}")
