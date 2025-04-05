# Airfare Price Prediction Project

This project aims to predict airfare prices using machine learning techniques. The model uses historical flight data to predict future airfare prices, helping users make informed booking decisions.

## Project Structure
```
airfare_prediction/
├── data/
│   └── Clean_Dataset.csv
├── notebooks/
│   └── exploratory_data_analysis.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── utils.py
├── models/
│   └── trained_models/
├── app/
│   ├── app.py
│   └── templates/
├── requirements.txt
└── README.md
```

## Features
- Data preprocessing and feature engineering
- Multiple ML models (Linear Regression, Random Forest, XGBoost)
- Model evaluation and comparison
- Interactive web interface using Streamlit
- Detailed documentation and analysis

## Installation
1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Run the Streamlit app:
```bash
streamlit run app/app.py
```

2. Access the web interface at http://localhost:8501

## Model Performance
- Linear Regression: R² Score, MSE
- Random Forest: R² Score, MSE
- XGBoost: R² Score, MSE

## Technologies Used
- Python
- Scikit-learn
- XGBoost
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Streamlit

## Contributing
Feel free to submit issues and enhancement requests! 