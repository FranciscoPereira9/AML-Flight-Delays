import pandas as pd
import numpy as np
import utils
import matplotlib.pyplot as plt
# ML
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)


# Read pre_proccesed data:
pp_train = pd.read_csv("../data/pp_flights_train.csv")
pp_test = pd.read_csv("../data/pp_flights_test.csv")
# Build Model
X, Y = pp_train['DEPARTURE_DELAY'].to_numpy().reshape(-1, 1) , pp_train['ARRIVAL_DELAY']
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.05, random_state=42)
X_test = pp_test['DEPARTURE_DELAY'].to_numpy().reshape(-1,1)
lm = LinearRegression().fit(X_train,Y_train)
# Check Validation Metrics
predictions_val = lm.predict(X_val)
print("MSE =", metrics.mean_squared_error(predictions_val, Y_val))
# Build prediction for Test Set
predictions_test = lm.predict(X_test)

# Visualize Results
# Plot model regression line on top of actual delays - Baseline
plt.scatter(X_val.reshape(-1), Y_val.to_numpy())
plt.plot(X_val.reshape(-1), predictions_val.reshape(-1), color='orange')
plt.show()
# Train Model

# Output Predictions
utils.create_output(predictions_test)
