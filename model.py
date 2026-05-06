import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def train_model():
    df = pd.read_csv("data.csv")

    X = df[['Open', 'High', 'Low']]
    y = df['Close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    # Save model
    joblib.dump(model, "model.pkl")

    return model, mse


def load_model():
    model = joblib.load("model.pkl")
    return model
