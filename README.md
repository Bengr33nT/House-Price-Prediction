# Housing Price Prediction

## Overview

This project implements a machine learning model to predict housing prices based on various features such as location, income, and housing characteristics. The dataset used is the California housing dataset, which is analyzed to train a Random Forest regression model.

## Features

- **Data Analysis**: Loads and explores the housing dataset.
- **Data Preprocessing**: Handles missing values and performs one-hot encoding for categorical variables.
- **Feature Engineering**: Creates new features like `bedroom_ratio` and `household_rooms`.
- **Model Training**: Utilizes a Random Forest regression model for predictions.
- **Evaluation**: Assesses model accuracy using R² score.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Seaborn
- Matplotlib

## Getting Started

### Prerequisites

- Python 3.x
- Required libraries: `pandas`, `scikit-learn`, `seaborn`, `matplotlib`.

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/housing-price-prediction.git
    cd housing-price-prediction
    ```

2. Install the required libraries:
    ```bash
    pip install pandas scikit-learn seaborn matplotlib
    ```

### Running the Analysis

1. Load the dataset:
    ```python
    import pandas as pd
    data = pd.read_csv("housing.csv")
    ```

2. Preprocess the data and create new features:
    ```python
    data.dropna(inplace=True)
    # Additional preprocessing steps...
    ```

3. Split the data into training and testing sets:
    ```python
    from sklearn.model_selection import train_test_split

    X = data.drop(['median_house_value'], axis=1)
    y = data['median_house_value']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    ```

4. Train the Random Forest model:
    ```python
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    ```

5. Evaluate the model's performance:
    ```python
    score = model.score(X_test, y_test)
    print(f"Model R² Score: {score}")
    ```

## Data Description

The dataset contains the following columns:

- `longitude`: Longitude of the house location.
- `latitude`: Latitude of the house location.
- `housing_median_age`: Median age of houses in the neighborhood.
- `total_rooms`: Total number of rooms.
- `total_bedrooms`: Total number of bedrooms.
- `population`: Population of the area.
- `households`: Number of households in the area.
- `median_income`: Median income of the area.
- `median_house_value`: Value of the house.
- `ocean_proximity`: Proximity to the ocean.

## Acknowledgments

This project utilizes the California housing dataset from the UCI Machine Learning Repository.
