#Solar Irradiance Forecasting Model
This repository contains a project focused on building a neural network model for forecasting solar irradiance based on meteorological data. The project includes data pre-processing steps, model training, performance analysis, and future plans for real-time irradiance reporting using Arduino.
#Project Overview
Accurate forecasting of solar irradiance is critical for efficient solar power generation and management. This project employs an Artificial Neural Network (ANN) to predict solar irradiance based on weather-related features such as temperature, humidity, and wind direction. Various model optimization techniques, including gradient descent, stochastic gradient descent (SGD), and gradient clipping, have been used to improve model accuracy and stability.
#Key Features
•	Data Pre-Processing: Extracted 16 features from meteorological data for model input, normalized the data, and split it into training and testing datasets.
•	Model Optimization:
•	Weight Initialization: Utilized Kaiming He initialization to mitigate gradient vanishing/exploding issues.
•	Gradient Descent Variants: Compared full-batch and mini-batch gradient descent methods for convergence.
•	Gradient Clipping: Limited gradient norms to control divergence, ensuring stable convergence.
•	Model Evaluation: Compared the ANN model against a linear regression model, with ANN demonstrating significantly lower error rates.
•	Real-Time Prediction (Future Scope): Planned implementation of real-time solar irradiance prediction using data from an Arduino kit.
**Results**
The neural network model achieved an MSE of approximately 8,752.26 on the testing data, significantly outperforming a linear regression model which achieved an MSE of 35,658.31.
