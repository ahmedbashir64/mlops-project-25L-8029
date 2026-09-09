import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import os

# Student ID: 25L-8029
LEARNING_RATE = 0.1  # added hyperparameter
def load_data(path="data/dataset.csv"):
    print("Loading dataset...")
    df = pd.read_csv(path)
    return df

def train_model(df, target_col="price"):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X = (X - X.mean()) / X.std()  # normalization step
    X = (X - X.min()) / (X.max() - X.min())  # min-max scaling
    X = pd.get_dummies(X, drop_first=True)  # handle categorical cols simply

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    print(f"Model R^2 score: {score:.4f}")
    return model

def save_model(model, path="model/model_25L-8029.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to {path}")

if __name__ == "__main__":
    df = load_data()
    model = train_model(df)
    save_model(model)