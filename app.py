import streamlit as st
import joblib
import pandas as pd
import google.generativeai as genai
import os
from dotenv import load_dotenv
import re

# Load .env file
load_dotenv()

# Get Gemini API key
api_key = os.getenv("GEMINI_API_KEY")

# Configure Gemini
genai.configure(api_key=api_key)

# =======================
# Load the trained model
# =======================
@st.cache_resource
def load_model():
    # Make sure price_predictor2.pkl is in the same directory as app.py
    return joblib.load("price_predictor.pkl")

model = load_model()

# =======================
# Function to get LLM-based explanation
# =======================
def get_llm_explanation(product_name, category, about_product, rating, ml_price, llm_price):
    prompt = f"""
    The ML model predicted ₹{ml_price:.2f} for a product. A separate AI estimated its price to be ₹{llm_price:.2f}.
    The product details are:
    - Name: {product_name}
    - Category: {category}
    - About: {about_product}
    - Rating: {rating}
    
    Please explain the difference between the two predicted prices.
    Explain why the ML model might have given a lower or higher prediction and why the LLM's price is a more realistic estimate for this product.
    """
    try:
        response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Could not generate explanation: {e}"

# =======================
# Function to get LLM-based price
# =======================
def get_llm_price(product_name, category, about_product):
    prompt = f"""
    Based on the following product details, provide a realistic price estimate in Indian Rupees (₹) as a single number. Do not include any text or symbols other than the number itself.
    
    Product Name: {product_name}
    Category: {category}
    About: {about_product}
    
    Example output: 1500.00
    """
    try:
        response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)
        price_match = re.search(r'₹?(\d[\d,\.]*)', response.text)
        if price_match:
            price_str = price_match.group(1).replace(',', '')
            return float(price_str)
        else:
            return None
    except Exception as e:
        return None

# =======================
# Streamlit UI
# =======================
st.set_page_config(page_title="E-Commerce Price Prediction", page_icon="🛒", layout="wide")
st.title("🛒 E-Commerce Price Prediction Tool")
st.write("Enter product details to get a predicted price and an AI-generated explanation.")

# Initialize session state for storing predictions
if "predictions" not in st.session_state:
    st.session_state.predictions = []

# --- Data for UI enhancements ---
product_data = {
    "Electronics": ["Smartphone", "Laptop", "Wireless Headphones", "Smartwatch"],
    "Toys & Games": ["Board Game", "Action Figure", "Building Blocks", "Doll House"],
    "Office Products": ["Fountain Pen", "Notebook", "Desk Organizer", "Ergonomic Chair"],
    "Home & Kitchen": ["Coffee Maker", "Blender", "Air Fryer", "Toaster"],
    "Clothing": ["T-Shirt", "Jeans", "Jacket", "Sweater"],
    "Books": ["Fantasy Novel", "Biography", "Cookbook", "Sci-Fi Book"],
    "Beauty & Personal Care": ["Face Wash", "Shampoo", "Lotion", "Perfume"],
    "Home Improvement": ["LED Light", "Drill Machine", "Water Purifier"],
    "Musical Instruments": ["Acoustic Guitar", "Digital Keyboard"]
}
categories = list(product_data.keys())

# --- Main app layout with columns ---
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Product Details")
    
    # User inputs
    category = st.selectbox("Product Category", options=categories)
    
    # New flexible product name input
    selected_product = st.selectbox("Or, select from a few common products:", options=[""] + product_data.get(category, []))
    product_name = st.text_input("Product Name", value=selected_product)
    
    about_product = st.text_area("About Product", height=100, value="Describe the product features here.")
    
    rating = st.slider("Rating (1-5)", min_value=0.0, max_value=5.0, value=4.0, step=0.1)
    
    rating_count = 0.0

    # Predict button
    if st.button("🔮 Predict Price"):
        if not category or not product_name:
            st.warning("⚠️ Please enter at least Product Name and Category.")
        else:
            input_data = pd.DataFrame([{
                "category": category,
                "product_name": product_name,
                "about_product": about_product,
                "rating": rating,
                "rating_count": rating_count
            }])

            try:
                # Make predictions from both models
                ml_prediction = model.predict(input_data)[0]
                llm_price = get_llm_price(product_name, category, about_product)

                # Display prices
                st.success(f"💰 ML Model Predicted Price: ₹{ml_prediction:,.2f}")
                if llm_price is not None:
                    st.info(f"✨ AI Price Estimate: ₹{llm_price:,.2f}")

                # Get and display the combined explanation
                explanation = get_llm_explanation(product_name, category, about_product, rating, ml_prediction, llm_price)
                st.info(f"🤖 Explanation:\n\n{explanation}")
                
                # Store prediction in session state
                st.session_state.predictions.append({
                    "Product": product_name,
                    "Category": category,
                    "ML Price": f"₹{ml_prediction:,.2f}",
                    "LLM Price": f"₹{llm_price:,.2f}" if llm_price is not None else "N/A",
                    "Rating": rating
                })
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

with col2:
    st.header("Previous Predictions")
    # Display previous predictions in a table
    if st.session_state.predictions:
        df_predictions = pd.DataFrame(st.session_state.predictions)
        st.table(df_predictions)
    else:
        st.info("No predictions yet. Your predictions will appear here.")
