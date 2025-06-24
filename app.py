import streamlit as st
import pandas as pd
from model import TitanicModel

# Page config
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide"
)

# Title and description
st.title("🚢 Titanic Survival Predictor")
st.write("""
### Would you have survived the Titanic?
This app predicts whether a passenger would survive the Titanic disaster based on their characteristics.
Enter your details below to find out!
""")

def load_model():
    model = TitanicModel()
    try:
        model.load_model("titanic_model.pkl")
    except FileNotFoundError:
        # If model doesn't exist, train it
        train_data = pd.read_csv("train.csv")
        model.train(train_data)
        model.save_model("titanic_model.pkl")
    return model

# Create the input form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        sex = st.selectbox("Sex", ["male", "female"])
        age = st.number_input("Age", min_value=0, max_value=100, value=30)
        embarked = st.selectbox(
            "Port of Embarkation", 
            ["S", "C", "Q"], 
            help="C = Cherbourg, Q = Queenstown, S = Southampton"
        )
    
    with col2:
        sibsp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
        parch = st.number_input("Number of Parents/Children Aboard", min_value=0, max_value=10, value=0)
        fare = st.number_input("Fare (in pounds)", min_value=0.0, max_value=600.0, value=32.0)

    submit_button = st.form_submit_button("Predict Survival")

# Make prediction when form is submitted
if submit_button:
    # Create a dataframe with the input data
    input_data = pd.DataFrame({
        'Sex': [sex],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare],
        'Embarked': [embarked]
    })
    
    # Load model and make prediction
    model = load_model()
    survival_proba = model.predict_proba(input_data)
    
    # Show prediction
    st.header("Prediction Results")
    
    # Create columns for probability display
    prob_col1, prob_col2 = st.columns(2)
    
    with prob_col1:
        st.metric(
            label="Survival Probability",
            value=f"{survival_proba[1]:.1%}"
        )
    with prob_col2:
        st.metric(
            label="Death Probability",
            value=f"{survival_proba[0]:.1%}"
        )
    
    # Display the interpretation
    if survival_proba[1] > 0.5:
        st.success(f"👍 You would likely have SURVIVED with a {survival_proba[1]:.1%} probability!")
    else:
        st.error(f"👎 You would likely NOT have survived with a {survival_proba[0]:.1%} probability!")
    
    # Show feature importance through passenger details
    st.subheader("Passenger Details Summary")
    st.write(f"- **Sex**: {sex}")
    st.write(f"- **Age**: {age}")
    st.write(f"- **Port of Embarkation**: {embarked}")
    st.write(f"- **Family Members Aboard**: {sibsp + parch}")
    st.write(f"- **Passenger Fare**: £{fare:.2f}")

# Add information about the model
with st.expander("About this app"):
    st.write("""
    This app uses a Random Forest Classifier trained on the Titanic dataset to predict passenger survival.
    The model takes into account various factors such as:
    - Passenger's sex
    - Age
    - Port of embarkation
    - Number of family members aboard (siblings, spouses, parents, children)
    - Ticket fare
    
    The model was trained on historical data from the Titanic disaster.
    """)
