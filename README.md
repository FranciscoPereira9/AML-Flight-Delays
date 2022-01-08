

# AML-Flight-Delays

Predicting flight delays - final project for Applied Machine Learning at University of Amsterdam.  

## Poster

 [NODE_FlightDelayPrediction.pdf](NODE_FlightDelayPrediction.pdf) 


## To Do

- [x] Poster

- Pre-processing:  (Nina)
  - [x] Drop columns that do not matter - TAIL_NUMBER, FLIGHT_NUMBER, WEELS_OFF, id
  - [x] Put YEAR, MONTH, DAY in → Datetime Object
  - [x] Deal with missing values
  - [x] One Hot Encode DAY_OF_WEEK,  AIRLINE 
  - [x] Target Encode categorical variables → ORIGIN_AIRPORT- DESTINATION_AIRPORT

- Modelling:
  - [x] Linear Regression between Departure Delay x Arrival Delay
  - [x] NODE Algorithm Running on our data 
  
- Thing to try:
	- [x] Adam Optimizer (did not improve)
	- [x] Autoencoder to calculate an input embedding space (did not improve)
	- [ ] Non-Oblivious Decision Trees (Asymetric)
	- [x] FCN (did not improve)

  ## Hyper-parameter Tuning

| Name                   | #Trees | Layers | Tree Dimension | Depth | MSE   | MSE Val |
| ---------------------- | ------ | ------ | -------------- | ----- | ----- | ------- |
| 2021_12_16-06_51_40_PM | 128    | 4      | 3              | 6     | 92.45 | 94.570  |
| 2021_12_17-02_18_31_PM | 128    | 6      | 3              | 6     | 92.24 | 91.006  |
| 2021_12_17-03_50_42_PM | 128    | 4      | 6              | 6     | 92.00 | 90.870  |
| 2021_12_18-03_07_10_PM | 256    | 8      | 5              | 6     | 94.29 | 96.33   |
| Encoded_NODES          | 256    | 8      | 5              | 6     | 96.32 | 98.23   |
| On ARRIVAL_DELAY       | 64     | 5      | 8              | 4     | 94.32 | 95.79   |

