# Used Car Price Prediction - Enhanced CatBoost Model

The original compeition site https://tianchi.aliyun.com/competition/entrance/231784


A machine learning project for predicting used car prices using CatBoost with advanced feature engineering. This project implements a GPU-accelerated CatBoost model with comprehensive feature engineering techniques to achieve competitive prediction accuracy.

## 📊 Project Overview

This project predicts used car prices based on various vehicle characteristics including brand, model, age, mileage, power, and other technical specifications. The model uses CatBoost gradient boosting algorithm with extensive feature engineering to capture complex patterns in the data.

### Performance Metrics

- **Current OOF MAE**: ~484.6 (5-fold cross-validation)

- Acutal Competition Score is

Date:2025-12-08 01:28:54
score:472.4529

- **Training Time**: ~5-10 minutes (GPU) / ~15-30 minutes (CPU)

## ✨ Key Features

### Advanced Feature Engineering

1. **Time Features**
   - Vehicle age (years and days)
   - Registration date features (year, month, day, season)
   - Creation date features
   - Relative age from current year
   - New car indicator (< 1 year)

2. **Missing Value Handling**
   - Missing value indicators for all numerical features
   - Median imputation with proper train/test separation
   - Missing value flags for: `power`, `kilometer`, `v_0` ~ `v_14`

3. **Outlier Detection & Treatment**
   - IQR-based outlier detection for `power`, `kilometer`, `v_0`
   - Outlier flags for model learning
   - Actual value clipping using training set statistics

4. **Statistical Features**
   - Brand-level statistics (mean, median, std, count)
   - Model-level statistics (mean, median, std, count)
   - Computed within CV folds to avoid data leakage
   - V-features statistics (mean, std, min, max, median)

5. **Categorical Encoding**
   - Frequency encoding for categorical features
   - Brand-model combinations
   - Model numerical encoding

6. **Interaction Features**
   - Power-displacement ratio (`power / v_0`)
   - Kilometer per year
   - Power per year
   - Power × age, Kilometer × age
   - Power + model combinations

## 🚀 Installation

### Prerequisites

- Python 3.7+
- GPU with CUDA support (optional, falls back to CPU if unavailable)

### Required Packages

```bash
pip install pandas numpy scikit-learn catboost
```

Or install from requirements file:

```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
tomversion/
├── train_fast_catboost_gpu.py          # Main training script (enhanced version)
├── train_fast_catboost_gpu_score_472.py # Alternative version
├── compare_predictions.py               # Comparison analysis script
├── IMPROVEMENT_PLAN.md                  # Improvement plan based on baseline comparison
├── used_car_train_20200313.csv          # Training data
├── used_car_testB_20200421.csv         # Test data
├── price_prediction_fast_catboost.csv   # Prediction output
├── catboost_info/                      # CatBoost training logs
└── README.md                           # This file
```

## 🎯 Usage

### Basic Usage

1. **Prepare Data**
   - Ensure `used_car_train_20200313.csv` and `used_car_testB_20200421.csv` are in the same directory as the script

2. **Run Training**
   ```bash
   python train_fast_catboost_gpu.py
   ```

3. **Output**
   - Predictions will be saved to `price_prediction_fast_catboost.csv`
   - Format: `SaleID, price`

## 🔧 Model Configuration

### CatBoost Parameters

```python
{
    'loss_function': 'MAE',
    'eval_metric': 'MAE',
    'depth': 7,
    'learning_rate': 0.03,
    'iterations': 2000,
    'l2_leaf_reg': 4.0,
    'random_seed': 42,
    'task_type': 'GPU'  # Auto-falls back to CPU if GPU unavailable
}
```

### Cross-Validation

- **Method**: 5-fold KFold
- **Shuffle**: True
- **Random State**: 42
- **Early Stopping**: 200 rounds

### Target Transformation

- **Training**: `log1p(y)` transformation
- **Prediction**: `expm1(pred)` inverse transformation
- **Post-processing**: Clipping to [200, 300000]

## 🔍 Feature Engineering Details

### Preprocessing Pipeline

1. **Date Parsing**: Convert `regDate` and `creatDate` to datetime
2. **Missing Value Indicators**: Create flags for all missing values
3. **Outlier Detection**: IQR-based detection and clipping
4. **Statistical Aggregation**: Brand/model level stats within CV folds
5. **Frequency Encoding**: Categorical feature frequency encoding
6. **Feature Combination**: Brand-model, power-model interactions

### Data Leakage Prevention

- Statistical features computed only on training folds
- Outlier clipping bounds from training set applied to test set
- Frequency encoding based on training set only

## 🎓 Model Architecture

### Training Process

1. **Data Loading**: Load train and test CSV files (space-separated)
2. **Feature Engineering**: Apply comprehensive preprocessing
3. **GPU Detection**: Attempt GPU training, fallback to CPU
4. **Cross-Validation**: 5-fold CV with early stopping
5. **Final Model**: Train on full dataset
6. **Prediction**: Generate test set predictions

### Output Format

```csv
SaleID,price
200000,1255.96
200001,2001.80
...
```

## 📝 Notes

- The model automatically detects and uses GPU if available
- All feature engineering is designed to prevent data leakage
- Predictions are clipped to reasonable bounds [200, 300000]
- The model uses log1p transformation for better handling of price distribution

## 🤝 Contributing

This is a learning project for used car price prediction. Suggestions and improvements are welcome!

## 📄 License

This project is for educational purposes.

## 👤 Author

Used Car Price Prediction Project - Enhanced CatBoost Implementation

---

**Last Updated**: December 2025

