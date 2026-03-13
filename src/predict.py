import joblib
import pandas as pd

# load trained model
model = joblib.load("sales_model.pkl")

print("Model loaded successfully!")

# sample input
sample = pd.DataFrame({
    "month":[6],
    "day":[15],
    "weekday":[2]
})

prediction = model.predict(sample)

print("Predicted Sales:", prediction)