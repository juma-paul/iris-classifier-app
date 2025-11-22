import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Iris Flower Classifier", page_icon="🌸", layout="wide")

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #4B0082;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4B0082;
        color: white;
        font-size: 1.2rem;
        padding: 0.75rem;
        border-radius: 10px;
    }
    .species-card {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🌸 Iris Flower Species Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predict the species of an Iris flower using machine learning</p>', unsafe_allow_html=True)

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.write("This app uses a K-Nearest Neighbors model to classify Iris flowers into three species:")
    st.write("- 🌺 **Setosa**")
    st.write("- 🌸 **Versicolor**")
    st.write("- 🌼 **Virginica**")
    st.divider()
    st.write("**Model Accuracy:** 93%")
    st.write("**Training Data:** 150 samples")

# Main content
st.subheader("📊 Enter Flower Measurements")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.0, 0.1, help="Length of the sepal in centimeters")
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.5, 0.1, help="Length of the petal in centimeters")

with col2:
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5, 0.1, help="Width of the sepal in centimeters")
    petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2, 0.1, help="Width of the petal in centimeters")

# Display current measurements
st.divider()
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Sepal Length", f"{sepal_length} cm")
col_b.metric("Sepal Width", f"{sepal_width} cm")
col_c.metric("Petal Length", f"{petal_length} cm")
col_d.metric("Petal Width", f"{petal_width} cm")

st.divider()

# Predict button
if st.button("🔮 Predict Species", type="primary"):
    with st.spinner("Making prediction..."):
        payload = {
            "sepal_length": sepal_length,
            "sepal_width": sepal_width,
            "petal_length": petal_length,
            "petal_width": petal_width
        }
        
        try:
            response = requests.post("http://localhost:8000/predict", json=payload)
            
            if response.status_code == 200:
                prediction = response.json()["prediction"]
                species_names = ["Setosa", "Versicolor", "Virginica"]
                species_emojis = ["🌺", "🌸", "🌼"]
                species_colors = ["#FFB6C1", "#DDA0DD", "#FFD700"]
                
                predicted_species = species_names[prediction]
                predicted_emoji = species_emojis[prediction]
                
                st.success(f"### Prediction Complete!")
                st.markdown(f"""
                    <div style='background-color: {species_colors[prediction]}; padding: 2rem; border-radius: 15px; text-align: center;'>
                        <h1 style='color: white; font-size: 3rem;'>{predicted_emoji}</h1>
                        <h2 style='color: white;'>{predicted_species}</h2>
                    </div>
                """, unsafe_allow_html=True)
                
                # Show confidence (placeholder - you can add actual probability from model)
                st.balloons()
                
            else:
                st.error("❌ Prediction failed. Please check your input and try again.")
        except Exception as e:
            st.error(f"❌ Error connecting to API: {str(e)}")

# Footer
st.divider()
st.caption("Built with Streamlit •")

# import streamlit as st
# import requests

# st.set_page_config(page_title="Iris Flower Classifier", page_icon="🌸", layout="wide")

# st.title("🌸 Iris Flower Species Classifier")
# st.write("Enter the flower measurements below to predict its species.")

# col1, col2 = st.columns(2)

# with col1:
#     sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.0)
#     petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.5)

# with col2:
#     sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5)
#     petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2)

# if st.button("Predict Species", type="primary"):
#     payload = {
#         "sepal_length": sepal_length,
#         "sepal_width": sepal_width,
#         "petal_length": petal_length,
#         "petal_width": petal_width
#     }
    
#     response = requests.post("http://localhost:8000/predict", json=payload)
    
#     if response.status_code == 200:
#         prediction = response.json()["prediction"]
#         species_names = ["Setosa", "Versicolor", "Virginica"]
#         st.success(f"Predicted Species: **{species_names[prediction]}** 🌺")
#     else:
#         st.error("Prediction failed. Please check your input.")
