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
# Function to get LLM-based explanation with fallback
# =======================
def get_llm_explanation(product_name, category, about_product, rating, ml_price, llm_price, is_rule_based=False):
    # Handle None case for llm_price
    if llm_price is None:
        llm_price_str = "N/A (could not estimate)"
        comparison_text = f"The ML model predicted ₹{ml_price:.2f} for this product. However, the AI price estimation failed to provide a reliable estimate."
    else:
        price_source = "rule-based estimate" if is_rule_based else "AI estimate"
        llm_price_str = f"₹{llm_price:.2f}"
        comparison_text = f"The ML model predicted ₹{ml_price:.2f} for a product. A separate {price_source} calculated its price to be ₹{llm_price:.2f}."
    
    # If quota exceeded, provide rule-based explanation
    if is_rule_based:
        return generate_rule_based_explanation(product_name, category, about_product, rating, ml_price, llm_price)
    
    prompt = f"""
    {comparison_text}
    The product details are:
    - Name: {product_name}
    - Category: {category}
    - About: {about_product}
    - Rating: {rating}
    
    Please explain the difference between the two predicted prices (if both are available).
    Explain why the ML model might have given a lower or higher prediction and provide insights about realistic pricing for this product.
    If the AI price estimation failed, focus on explaining the ML model's prediction and what factors might influence the actual market price.
    """
    try:
        response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)
        return response.text
    except Exception as e:
        if "429" in str(e) or "quota" in str(e).lower():
            return generate_rule_based_explanation(product_name, category, about_product, rating, ml_price, llm_price)
        return f"⚠️ Could not generate explanation: {e}"

# =======================
# Rule-based explanation fallback
# =======================
def generate_rule_based_explanation(product_name, category, about_product, rating, ml_price, llm_price):
    """Generate explanation when LLM is unavailable"""
    explanation = f"""
    **Price Analysis (Rule-based fallback):**
    
    The ML model predicted ₹{ml_price:.2f} while the rule-based estimate is ₹{llm_price:.2f}.
    
    **Factors influencing the pricing:**
    
    1. **Category Impact**: {category} products typically have varying price ranges based on features and brand positioning.
    
    2. **Rating Influence**: With a {rating}/5 rating, this product {"appears to be well-received" if rating >= 4 else "may have some quality concerns" if rating < 3 else "has average market reception"}.
    
    3. **Price Difference Analysis**:
    """
    
    price_diff = abs(ml_price - llm_price)
    if price_diff < ml_price * 0.1:  # Less than 10% difference
        explanation += "   - Both estimates are quite close, suggesting consistent market positioning."
    elif ml_price > llm_price:
        explanation += f"   - The ML model estimates ₹{price_diff:.2f} higher, possibly due to specific feature combinations or brand factors in its training data."
    else:
        explanation += f"   - The rule-based estimate is ₹{price_diff:.2f} higher, possibly because it uses category averages rather than specific product features."
    
    explanation += f"""
    
    4. **Market Factors**: Consider factors like brand reputation, specific features mentioned in the product description, and current market trends when making final pricing decisions.
    
    *Note: This analysis uses rule-based logic due to API limitations. For more detailed insights, try again when the API quota resets.*
    """
    
    return explanation

# =======================
# Function to get LLM-based price with fallback
# =======================
def get_llm_price(product_name, category, about_product):
    # First try rule-based pricing as fallback
    fallback_price = get_rule_based_price(product_name, category, about_product)
    
    prompt = f"""
    Based on the following product details, provide a realistic price estimate in Indian Rupees (₹) as a single number. Do not include any text or symbols other than the number itself.
    
    Product Name: {product_name}
    Category: {category}
    About: {about_product}
    
    Example output: 1500.00
    """
    try:
        response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)
        # Improved regex to catch various number formats
        price_match = re.search(r'₹?\s*(\d+(?:[,\.]\d+)*(?:\.\d{2})?)', response.text.strip())
        if price_match:
            price_str = price_match.group(1).replace(',', '')
            return float(price_str)
        else:
            # Try to find any number in the response
            number_match = re.search(r'(\d+(?:\.\d+)?)', response.text)
            if number_match:
                return float(number_match.group(1))
            return fallback_price
    except Exception as e:
        if "429" in str(e) or "quota" in str(e).lower():
            st.warning("⚠️ AI quota exceeded. Using rule-based pricing estimate.")
            return fallback_price
        else:
            st.error(f"Error in LLM price estimation: {e}")
            return fallback_price

# =======================
# Rule-based pricing fallback
# =======================
def get_rule_based_price(product_name, category, about_product):
    """Simple rule-based pricing when LLM is unavailable"""
    base_prices = {
        "Electronics": 15000,
        "Toys & Games": 800,
        "Office Products": 500,
        "Home & Kitchen": 3000,
        "Clothing": 1200,
        "Books": 400,
        "Beauty & Personal Care": 600,
        "Home Improvement": 2000,
        "Musical Instruments": 8000
    }
    
    base_price = base_prices.get(category, 1000)
    
    # Adjust based on product name keywords
    product_lower = product_name.lower()
    multipliers = {
        'premium': 2.0,
        'pro': 1.8,
        'deluxe': 1.6,
        'luxury': 2.5,
        'professional': 1.7,
        'smart': 1.4,
        'wireless': 1.2,
        'bluetooth': 1.3,
        'digital': 1.2,
        'automatic': 1.3,
        'basic': 0.7,
        'simple': 0.8,
        'mini': 0.6,
        'portable': 0.9
    }
    
    multiplier = 1.0
    for keyword, mult in multipliers.items():
        if keyword in product_lower:
            multiplier *= mult
    
    # Adjust based on about_product keywords
    about_lower = about_product.lower()
    if any(word in about_lower for word in ['high-end', 'premium', 'luxury']):
        multiplier *= 1.5
    elif any(word in about_lower for word in ['budget', 'affordable', 'cheap']):
        multiplier *= 0.7
    
    return round(base_price * multiplier, 2)

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
                else:
                    st.warning("⚠️ AI Price Estimate: Could not generate estimate")

                # Get and display the combined explanation (now handles None llm_price)
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
