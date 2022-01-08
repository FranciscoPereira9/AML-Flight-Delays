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

filename = "2021_12_19-03_06_40_PM"
# Read pre_proccesed data:
flights_train = pd.read_csv("data/flights_train.csv")
flights_test = pd.read_csv("data/flights_test.csv")
pp_train = pd.read_csv("data/pp_flights_train.csv")
pp_test = pd.read_csv("data/pp_flights_test.csv")
submission = pd.read_csv("data/residuals/"+filename+".csv")
# Fit Linear Model
X, Y = pp_train['DEPARTURE_DELAY'].to_numpy().reshape(-1, 1), pp_train['ARRIVAL_DELAY']
X_test = pp_test['DEPARTURE_DELAY'].to_numpy().reshape(-1, 1)
lm = LinearRegression().fit(X, Y)
# Build prediction for Test Set
flights_test["LM_PREDICTIONS"] = lm.predict(X_test)
submission["ARRIVAL_DELAY"] = flights_test["LM_PREDICTIONS"]+submission["RESIDUALS"]
submission.drop(columns="RESIDUALS", inplace=True)
# Visualize Results
# Plot model regression line on top of actual delays - Baseline
plt.scatter(submission.index.to_numpy(), submission.ARRIVAL_DELAY.to_numpy())
plt.plot(submission.index.to_numpy(), flights_test.LM_PREDICTIONS.to_numpy(), color='orange')
plt.show()
# To csv
submission.to_csv("submissions/"+filename+".csv", index=False)
print("DONE")