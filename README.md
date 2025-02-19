# Boston Housing Prices Prediction Project

## Overview

This project predicts housing prices in Boston based on various features using **linear regression**. The implementation includes data preprocessing, exploratory data analysis, model training, evaluation, and gradient descent optimization.

## Dataset

The dataset used is the **Boston Housing dataset**, which contains information about housing in Boston suburbs, including:

- **CRIM**: Per capita crime rate by town.
- **ZN**: Proportion of residential land zoned for lots over 25,000 sq. ft.
- **INDUS**: Proportion of non-retail business acres per town.
- **CHAS**: Charles River dummy variable (1 if tract bounds river; 0 otherwise).
- **NOX**: Nitric oxide concentration (parts per 10 million).
- **RM**: Average number of rooms per dwelling.
- **AGE**: Proportion of owner-occupied units built before 1940.
- **DIS**: Weighted distances to five Boston employment centers.
- **RAD**: Index of accessibility to radial highways.
- **TAX**: Full-value property-tax rate per \$10,000.
- **PTRATIO**: Pupil-teacher ratio by town.
- **B**: Proportion of Black population by town.
- **LSTAT**: Percentage of lower-status population.
- **MEDV**: Median value of owner-occupied homes (\$1000s).

## Features

- **Data Preprocessing**: Handling missing values, standardization, and feature selection.
- **Exploratory Data Analysis (EDA)**: Visualizing correlations and distributions.
- **Model Training**: Implementing Linear Regression using **Scikit-Learn**.
- **Performance Evaluation**: Using RMSE, R²-score, and error metrics.
- **Gradient Descent Optimization**: Improving model performance using optimization techniques.

## Installation

To run this project, install the required dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Usage

1. **Load the Dataset**: The dataset is available in the notebook or can be downloaded.
2. **Preprocess Data**: Handle missing values and standardize features.
3. **Train the Model**: Run the Jupyter Notebook to train the Linear Regression model.
4. **Evaluate Performance**: Use RMSE and R²-score for assessment.
5. **Optimize Using Gradient Descent**: Improve model predictions.

## Project Structure

```
Boston_Housing/
├── dataset/              # Housing dataset (if available separately)
├── models/               # Trained models
├── MDS_FINAL_Project2.ipynb  # Jupyter Notebook with implementation
├── README.md             # Project documentation
└── requirements.txt      # Dependencies
```

## Future Enhancements

- Experiment with different regression models (e.g., Ridge, Lasso, Decision Trees).
- Implement feature engineering techniques to improve accuracy.
- Deploy the model using Flask or FastAPI for real-time predictions.

## Contributors

- **Your Name** - Sara Elwakeel , Sara Aly

## License

This project is open-source and available under the MIT License.

