import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# Load dataset
df = pd.read_csv("train.csv")

print(df.columns)  # just to confirm columns

# Convert Order Date
df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)

# Feature engineering
df["month"] = df["Order Date"].dt.month
df["day"] = df["Order Date"].dt.day
df["weekday"] = df["Order Date"].dt.weekday

# Features for model
X = df[["month", "day", "weekday"]]

# Target
y = df["Sales"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Predictions
pred = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, pred)
print("Model MAE:", mae)

# Save model
joblib.dump(model, "sales_model.pkl")

print("Model saved successfully!")