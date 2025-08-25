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
    # Make sure price_predictor.pkl is in the same directory as app.py
    return joblib.load("price_predictor.pkl")

model = load_model()

# =======================
# Function to get LLM-based explanation
# =======================
def get_llm_explanation(product_name, category, about_product, rating, ml_price, llm_price):
    # Handle None values for llm_price
    llm_price_text = f"₹{llm_price:.2f}" if llm_price is not None else "unavailable"
    
    prompt = f"""
    The ML model predicted ₹{ml_price:.2f} for a product. A separate AI estimated its price to be {llm_price_text}.
    The product details are:
    - Name: {product_name}
    - Category: {category}
    - About: {about_product}
    - Rating: {rating}
    
    Please explain the difference between the two predicted prices.
    Explain why the ML model might have given a lower or higher prediction and why the LLM's price is a more realistic estimate for this product.
    If the LLM price is unavailable, focus on explaining the ML model's prediction based on the product details.
    """
    try:
        response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)
        return response.text
    except Exception as e:
        error_str = str(e)
        if "429" in error_str or "quota" in error_str.lower():
            return get_fallback_explanation(product_name, category, about_product, rating, ml_price, llm_price)
        return f"⚠️ Could not generate explanation: {e}"

# =======================
# Fallback explanation when API quota is exceeded
# =======================
def get_fallback_explanation(product_name, category, about_product, rating, ml_price, llm_price):
    """Provide a basic explanation when API quota is exceeded"""
    
    # Basic category-based price ranges (in INR)
    category_ranges = {
        "Electronics": {"min": 1000, "max": 100000},
        "Toys & Games": {"min": 200, "max": 5000},
        "Office Products": {"min": 100, "max": 20000},
        "Home & Kitchen": {"min": 500, "max": 50000},
        "Clothing": {"min": 300, "max": 10000},
        "Books": {"min": 100, "max": 2000},
        "Beauty & Personal Care": {"min": 200, "max": 5000},
        "Home Improvement": {"min": 500, "max": 25000},
        "Musical Instruments": {"min": 2000, "max": 50000}
    }
    
    explanation = f"🤖 **Analysis of ML Prediction (API quota exceeded, using basic analysis):**\n\n"
    explanation += f"**Predicted Price:** ₹{ml_price:.2f}\n\n"
    
    # Category analysis
    if category in category_ranges:
        cat_range = category_ranges[category]
        if ml_price < cat_range["min"]:
            explanation += f"📊 **Category Analysis:** The predicted price seems low for the {category} category, which typically ranges from ₹{cat_range['min']:,} to ₹{cat_range['max']:,}.\n\n"
        elif ml_price > cat_range["max"]:
            explanation += f"📊 **Category Analysis:** The predicted price is on the higher end for the {category} category, suggesting premium features.\n\n"
        else:
            explanation += f"📊 **Category Analysis:** The predicted price falls within the typical range for {category} products (₹{cat_range['min']:,} - ₹{cat_range['max']:,}).\n\n"
    
    # Rating analysis
    if rating >= 4.5:
        explanation += f"⭐ **Rating Impact:** High rating ({rating}/5) suggests good quality, which supports the predicted price.\n\n"
    elif rating >= 3.5:
        explanation += f"⭐ **Rating Impact:** Average rating ({rating}/5) indicates standard quality product.\n\n"
    else:
        explanation += f"⭐ **Rating Impact:** Lower rating ({rating}/5) might indicate budget pricing or quality concerns.\n\n"
    
    # Product name analysis
    if any(word in product_name.lower() for word in ['premium', 'pro', 'deluxe', 'advanced']):
        explanation += f"🏷️ **Product Analysis:** The name '{product_name}' suggests premium features, which may justify higher pricing.\n\n"
    elif any(word in product_name.lower() for word in ['basic', 'budget', 'simple', 'mini']):
        explanation += f"🏷️ **Product Analysis:** The name '{product_name}' suggests a budget or basic model, which aligns with competitive pricing.\n\n"
    
    if llm_price is not None:
        explanation += f"💡 **Comparison:** The AI estimated ₹{llm_price:.2f}, showing a {'higher' if llm_price > ml_price else 'lower'} valuation than the ML model.\n\n"
    
    explanation += "**Note:** This is a basic analysis due to API quota limits. For detailed AI insights, please try again after the quota resets."
    
    return explanation

# =======================
# Function to get LLM-based price
# =======================
def get_llm_price(product_name, category, about_product):
    prompt = f"""
    Based on the following product details, provide a realistic price estimate in Indian Rupees (₹). 
    Please respond with only a number (no currency symbols, no text, just the numeric value).
    
    Product Name: {product_name}
    Category: {category}
    About: {about_product}
    
    Example output format: 1500.00
    """
    try:
        response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)
        # More robust price extraction
        price_text = response.text.strip()
        
        # Try multiple regex patterns to extract price
        patterns = [
            r'₹?(\d+(?:[,\.]\d+)*(?:\.\d{2})?)',  # Match with or without ₹, with commas/dots
            r'(\d+(?:\.\d+)?)',  # Simple decimal number
            r'(\d+)'  # Just digits
        ]
        
        for pattern in patterns:
            price_match = re.search(pattern, price_text)
            if price_match:
                price_str = price_match.group(1).replace(',', '')
                try:
                    return float(price_str)
                except ValueError:
                    continue
        
        # If no pattern matches, try to convert the entire response
        try:
            return float(price_text.replace('₹', '').replace(',', '').strip())
        except:
            return None
            
    except Exception as e:
        error_str = str(e)
        if "429" in error_str or "quota" in error_str.lower():
            # Return a basic estimate when quota is exceeded
            return get_fallback_price(product_name, category, about_product)
        st.warning(f"Error getting LLM price: {e}")
        return None

# =======================
# Fallback price estimation when API quota is exceeded
# =======================
def get_fallback_price(product_name, category, about_product):
    """Provide a basic price estimate when API quota is exceeded"""
    
    # Basic category-based price ranges (in INR) - midpoint estimates
    category_estimates = {
        "Electronics": 15000,
        "Toys & Games": 1500,
        "Office Products": 2000,
        "Home & Kitchen": 5000,
        "Clothing": 1500,
        "Books": 500,
        "Beauty & Personal Care": 800,
        "Home Improvement": 3000,
        "Musical Instruments": 10000
    }
    
    base_price = category_estimates.get(category, 1000)
    
    # Adjust based on product name keywords
    name_lower = product_name.lower()
    
    # Premium indicators
    if any(word in name_lower for word in ['premium', 'pro', 'deluxe', 'advanced', 'professional']):
        base_price *= 2
    elif any(word in name_lower for word in ['luxury', 'high-end', 'flagship']):
        base_price *= 3
    # Budget indicators
    elif any(word in name_lower for word in ['basic', 'budget', 'simple', 'mini', 'compact']):
        base_price *= 0.6
    
    # Brand-like adjustments (if product name suggests brand)
    if any(word in name_lower for word in ['apple', 'samsung', 'sony', 'nike', 'adidas']):
        base_price *= 1.5
    
    return round(base_price, 2)

# =======================
# Streamlit UI
# =======================
st.set_page_config(page_title="E-Commerce Price Prediction", page_icon="🛒", layout="wide")
st.title("🛒 E-Commerce Price Prediction Tool")
st.write("Enter product details to get a predicted price and an AI-generated explanation.")

# Add info about API limits
with st.expander("ℹ️ About API Limits"):
    st.info("""
    **Gemini API Quota Information:**
    - Free tier: 50 requests per day
    - When quota is exceeded, the app uses fallback estimates
    - Quota resets every 24 hours
    - For unlimited access, consider upgrading your Gemini API plan
    """)

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
                # Make prediction from ML model
                ml_prediction = model.predict(input_data)[0]
                st.success(f"💰 ML Model Predicted Price: ₹{ml_prediction:,.2f}")
                
                # Get LLM price with proper error handling
                with st.spinner("Getting AI price estimate..."):
                    llm_price = get_llm_price(product_name, category, about_product)

                # Display LLM price with proper None handling
                if llm_price is not None:
                    # Check if this is a fallback estimate
                    if hasattr(st.session_state, 'using_fallback_price'):
                        st.info(f"💡 Fallback Price Estimate: ₹{llm_price:,.2f} (API quota exceeded)")
                    else:
                        st.info(f"✨ AI Price Estimate: ₹{llm_price:,.2f}")
                    llm_price_display = f"₹{llm_price:,.2f}"
                else:
                    st.warning("⚠️ Could not get AI price estimate")
                    llm_price_display = "N/A"

                # Get and display the combined explanation
                with st.spinner("Generating explanation..."):
                    explanation = get_llm_explanation(product_name, category, about_product, rating, ml_prediction, llm_price)
                    st.info(f"🤖 Explanation:\n\n{explanation}")
                
                # Store prediction in session state
                st.session_state.predictions.append({
                    "Product": product_name,
                    "Category": category,
                    "ML Price": f"₹{ml_prediction:,.2f}",
                    "LLM Price": llm_price_display,
                    "Rating": rating
                })
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                # Add more detailed error information for debugging
                import traceback
                st.error(f"Detailed error: {traceback.format_exc()}")

with col2:
    st.header("Previous Predictions")
    # Add a clear button for predictions
    if st.session_state.predictions and st.button("Clear History"):
        st.session_state.predictions = []
        st.rerun()
    
    # Display previous predictions in a table
    if st.session_state.predictions:
        df_predictions = pd.DataFrame(st.session_state.predictions)
        st.table(df_predictions)
    else:
        st.info("No predictions yet. Your predictions will appear here.")
