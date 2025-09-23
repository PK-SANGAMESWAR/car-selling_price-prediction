# Car Price Prediction Using Regression

A comprehensive machine learning project that predicts used car prices using various regression techniques including Linear Regression, Ridge Regression, and Lasso Regression.

## 📊 Project Overview

This project analyzes used car data to predict car prices based on features like car name, company, year, kilometers driven, and fuel type. The project implements and compares three different regression models to find the best approach for price prediction.

## 🚗 Dataset

The dataset contains information about used cars with the following features:
- **Name**: Car model name
- **Company**: Car manufacturer
- **Year**: Manufacturing year
- **Price**: Car price (target variable)
- **Kilometers Driven**: Total distance covered
- **Fuel Type**: Type of fuel (Petrol/Diesel/LPG)

## 🛠️ Technologies Used

- **Python 3.12+**
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **scikit-learn**: Machine learning algorithms
- **matplotlib**: Data visualization
- **uv**: Modern Python package management

## 📁 Project Structure

```
CAR_PRICE_PREDICTION_USING_REGRESSION/
│
├── carprediction.ipynb     # Main Jupyter notebook with analysis
├── main.py                 # Python script version
├── requirements.txt        # Project dependencies
├── pyproject.toml         # Project configuration
├── cleaned Car.csv        # Processed dataset
└── README.md              # Project documentation
```

## 🚀 Getting Started

### Prerequisites

Make sure you have Python 3.12+ installed on your system.

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/car-price-prediction-regression.git
   cd car-price-prediction-regression
   ```

2. **Install uv (if not already installed)**
   ```bash
   pip install uv
   ```

3. **Install dependencies**
   ```bash
   uv sync
   ```

   Or using pip:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Jupyter notebook**
   ```bash
   jupyter notebook carprediction.ipynb
   ```

## 🔍 Data Processing

The project includes comprehensive data cleaning steps:

1. **Data Quality Issues Identified:**
   - Year column contains non-numeric values
   - Price column has "Ask For Price" entries
   - Kilometers driven contains inconsistent formats
   - Missing values in fuel_type column

2. **Cleaning Steps:**
   - Filter out non-numeric year values
   - Remove "Ask For Price" entries
   - Standardize kilometers driven format
   - Handle missing values
   - Extract first 3 words from car names for consistency
   - Remove price outliers (> 6M)

## 🤖 Machine Learning Models

### 1. Linear Regression
- **Description**: Basic linear regression without regularization
- **Use Case**: Baseline model for comparison
- **Features**: Uses all available features

### 2. Ridge Regression
- **Description**: Linear regression with L2 regularization
- **Regularization**: Prevents overfitting by shrinking coefficients
- **Alpha Values Tested**: [0.1, 1.0, 10.0, 100.0]
- **Advantage**: Keeps all features but reduces their impact

### 3. Lasso Regression
- **Description**: Linear regression with L1 regularization
- **Feature Selection**: Automatically selects relevant features by setting coefficients to zero
- **Alpha Values Tested**: [0.1, 1.0, 10.0, 100.0, 1000.0]
- **Advantage**: Provides feature selection and model interpretability

## 📈 Model Performance

The models are evaluated using:
- **R² Score**: Coefficient of determination
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **MSE**: Mean Square Error

### Key Findings
- Model performance comparison across different regularization strengths
- Feature importance analysis
- Identification of most influential car characteristics for pricing


## 🎯 Key Insights

- **Feature Selection**: Lasso regression identifies the most important features for car price prediction
- **Regularization Impact**: Ridge and Lasso help prevent overfitting compared to basic linear regression
- **Model Interpretability**: The analysis provides insights into which car characteristics most influence pricing

## 🔧 Usage Example

```python
# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score

# Load and preprocess data
car = pd.read_csv('cleaned Car.csv')
X = car.drop(columns='Price')
y = car['Price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train Ridge regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

# Make predictions
y_pred = ridge.predict(X_test)
r2 = r2_score(y_test, y_pred)

print(f"R² Score: {r2:.4f}")
```

## 📋 Requirements

See `requirements.txt` for a complete list of dependencies. Main packages include:

```
scikit-learn==1.7.2
pandas==2.2.3
numpy==2.2.0
matplotlib==3.10.0
jupyter==1.1.1
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/PK-SANGAMESWAR)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/pk-sangameswar)

## 🙏 Acknowledgments

- Dataset source: [Quikr Car Dataset]
- Inspiration from various car price prediction projects
- Thanks to the open-source community for excellent ML libraries

## 📞 Support

If you have any questions or run into issues, please open an issue on GitHub or contact me directly.

---

⭐ **If you found this project helpful, please give it a star!** ⭐
