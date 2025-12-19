import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# load data
data = pd.read_csv("winequality-red.csv",sep=';')

# Feature and target split
X = data.drop("quality", axis=1)
y = data["quality"]

# build model and train
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(X, y)

# save the model
joblib.dump(model, "my_wine_model.joblib")
print("Model saved as my_wine_model.joblib")
