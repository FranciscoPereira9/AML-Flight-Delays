

# AML-Flight-Delays

Predicting flight delays - final project for Applied Machine Learning at University of Amsterdam.  

## To Do

- Assumption: Correlation and Causation between departure delay and arrival delay.
- Pre-processing: 
  - [x] Drop columns that do not matter - TAIL_NUMBER, FLIGHT_NUMBER, WEELS_OFF, id
  - [x] Put YEAR, MONTH, DAY in → Datetime Object
  - [x] Encode DAY_OF_WEEK and DEPARTURE_TIME
  - [x] Encode categorical variables → ORIGIN_AIRPORT, DESTINATION_AIRPORT, AIRLINE 
  - [x] Deal with missing values
- Modelling:
  - [ ] Linear Regression between Departure Delay x Arrival Delay
  - [ ] Deep Learning Regression
  - [ ] Gradient Boosting (?)
