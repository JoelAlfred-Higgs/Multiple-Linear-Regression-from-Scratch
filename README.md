# Multiple Linear Regression from Scratch

A simple Python project that implements multiple linear regression from scratch to predict house prices using a dataset with multiple features.

## Overview

This project trains a multiple linear regression model using:
- Area
- Bedrooms
- Bathrooms

It uses gradient descent to minimize the cost function and predict house prices based on user input.

## Project Structure

- `LinearRegression.py` - Main implementation of the regression model
- `House Price Prediction Dataset.csv` - Sample housing dataset
- `requirements.txt` - Python dependencies

## Dataset

The dataset contains house-related variables such as:

- `Area`
- `Bedrooms`
- `Bathrooms`
- `Floors`
- `YearBuilt`
- `Location`
- `Condition`
- `Garage`
- `Price`

The model uses the following features for training:
- `Area`
- `Bedrooms`
- `Bathrooms`

## Requirements

Make sure you have Python 3 installed, then install the required packages:

```bash
pip install -r requirements.txt
```

Required packages:
- numpy
- pandas
- matplotlib

## How to Run

From the root directory, run:

```bash
python LinearRegression.py
```

The script will prompt you for:

- House area
- Number of bedrooms
- Number of bathrooms

Then it will:
1. Train the model using gradient descent
2. Predict the house price
3. Display the cost graph

## Implementation Details

The model includes:

- Feature normalization
- Cost computation
- Gradient calculation
- Gradient descent optimization
- Prediction function

The model is implemented manually instead of using a high-level library like scikit-learn.

## Important Note

The script currently reads the dataset using an absolute Windows path:

```python
pd.read_csv(r"C:\Users\joela\OneDrive\Documents\Regression\House Price Prediction Dataset.csv")
```

If you are running this on another machine, update the path to the local dataset file, for example:

```python
data = pd.read_csv("House Price Prediction Dataset.csv")
```

This is necessary for the project to run correctly on other systems.

## Example Usage

```bash
Enter area of the House: 2500
Enter no of bedrooms: 3
Enter no of bathrooms: 2
```

The script will output a predicted price based on the trained model.

## License

This project is provided for educational purposes.

## Author

Joel Alfred Higgs
