import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

class TitanicModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.feature_names = None  # Store feature names used during training
        
    def preprocess_data(self, df):
        # Create a copy of the dataframe
        df = df.copy()
        
        # Drop unnecessary columns
        drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']
        df = df.drop(columns=[col for col in df.columns if col in drop_cols])
        
        # Fill missing values
        df['Age'] = df['Age'].fillna(df['Age'].median())
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())
        
        # Feature engineering
        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
        df['FamilySize'] = df['SibSp'] + df['Parch']
        
        # Create dummy variables for Embarked
        df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
        
        return df
    
    def train(self, train_data):
        # Preprocess the training data
        processed_data = self.preprocess_data(train_data)
        
        # Prepare features and target
        X = processed_data.drop('Survived', axis=1)
        y = processed_data['Survived']
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Split the data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        return self.model.score(X_val, y_val)
    
    def predict_proba(self, data):
        # Preprocess the input data
        processed_data = self.preprocess_data(data)
        
        # Ensure all required columns are present
        if self.feature_names is None:
            raise ValueError("Model is not trained or feature names are missing.")
        for col in self.feature_names:
            if col not in processed_data.columns:
                processed_data[col] = 0
        processed_data = processed_data[self.feature_names]
        
        # Make prediction
        prediction_proba = self.model.predict_proba(processed_data)
        
        return prediction_proba[0]
    
    def predict(self, data):
        processed_data = self.preprocess_data(data)
        if self.feature_names is None:
            raise ValueError("Model is not trained or feature names are missing.")
        for col in self.feature_names:
            if col not in processed_data.columns:
                processed_data[col] = 0
        processed_data = processed_data[self.feature_names]
        return self.model.predict(processed_data)[0]
    
    def save_model(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({'model': self.model, 'feature_names': self.feature_names}, f)
    
    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.feature_names = data['feature_names']

if __name__ == "__main__":
    # Load the training data
    train_data = pd.read_csv("train.csv")
    
    # Initialize and train the model
    model = TitanicModel()
    accuracy = model.train(train_data)
    print(f"Model validation accuracy: {accuracy:.2f}")
    
    # Save the trained model
    model.save_model("titanic_model.pkl")
