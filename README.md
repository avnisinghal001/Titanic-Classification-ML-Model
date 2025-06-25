# Titanic Survival Prediction - Machine Learning Project

## Project Overview
This project implements a machine learning model to predict the survival probability of Titanic passengers using the famous Titanic dataset. The project includes a Random Forest classifier model and a user-friendly Streamlit web interface for making predictions.

## Problem Statement
The sinking of the Titanic is one of the most infamous shipwrecks in history. On April 15, 1912, during her maiden voyage, the widely considered "unsinkable" RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren't enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.

This project aims to answer the question: "What sorts of people were more likely to survive?" using passenger data (name, age, gender, socio-economic class, etc.).

## Solution Approach

### Data Preprocessing
- Handling missing values in Age, Embarked, and Fare columns
- Feature engineering: Creating FamilySize feature
- Converting categorical variables (Sex, Embarked) into numerical format
- Dropping unnecessary columns (PassengerId, Name, Ticket, Cabin)

### Machine Learning Model
- Algorithm: Random Forest Classifier
- Features used:
  - Pclass (Passenger Class)
  - Sex (Gender)
  - Age
  - SibSp (Siblings/Spouses Aboard)
  - Parch (Parents/Children Aboard)
  - Fare
  - Embarked (Port of Embarkation)
  - FamilySize (Derived feature)

### Model Performance
- Validation accuracy: ~80-85% (may vary slightly due to random split)
- Uses cross-validation to ensure robust performance

## Project Structure
```
├── app.py              # Streamlit web application
├── model.py            # Machine learning model implementation
├── requirements.txt    # Project dependencies
├── train.csv          # Training dataset
├── test.csv           # Test dataset
└── titanic_model.pkl  # Saved trained model
```

## Installation & Setup

1. Clone the repository:
```bash
git clone [repository-url]
cd Titanic-Classification-ML-Model
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Train the model:
```bash
python model.py
```

5. Run the Streamlit app:
```bash
streamlit run app.py
```

## Docker Deployment

1. Build the Docker image:
```bash
docker build -t titanic-streamlit-app .
```

2. Run the container:
```bash
docker run -p 8501:8501 titanic-streamlit-app
```

## Cloud Deployment (Google Cloud)

1. Push to Google Container Registry:
```bash
docker tag titanic-streamlit-app gcr.io/[PROJECT-ID]/titanic-streamlit-app
docker push gcr.io/[PROJECT-ID]/titanic-streamlit-app
```

2. Deploy to Cloud Run:
- Go to Google Cloud Console
- Navigate to Cloud Run
- Click "Create Service"
- Select the container image
- Configure the service settings
- Deploy

## Usage
1. Access the web interface (locally at http://localhost:8501)
2. Input passenger details:
   - Passenger Class (1st, 2nd, or 3rd)
   - Gender
   - Age
   - Number of Siblings/Spouses
   - Number of Parents/Children
   - Fare
   - Port of Embarkation
3. Click "Predict" to see survival probability

## Technologies Used
- Python 3.12
- Scikit-learn
- Pandas
- Streamlit
- Docker
- Google Cloud Platform (for deployment)

## Future Improvements
- Feature importance analysis
- Model comparison (try other algorithms)
- Hyperparameter tuning
- Additional feature engineering
- Enhanced UI/UX
- API endpoint creation

## Contributing
Feel free to fork the project and submit pull requests for any improvements.

## License
This project is open-sourced under the MIT license.
