# 🧠 Brain-Age Prediction Project

**Advanced Machine Learning 2025 - Project 1**

---

## 🏆 Best Model Performance

- **CV R²:** 0.5076
- **Model:** XGBoost (80%) + SVR (20%) Ensemble
- **Feature Selection:** ReliefF (k=250)
- **Overfitting:** 18.6%

---

## 📁 Quick Navigation

```
.
├── final_models/              ⭐ BEST MODELS - START HERE
├── experiments/               🔬 Feature selection & tuning
├── analysis/                  📊 Results & visualizations  
├── utilities/                 🛠 Helper modules
├── documentation/             📚 Project docs
├── archived/                  🗄 Old experiments
└── eth-aml-2025-project-1/   📊 Dataset
```

**👉 See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for complete documentation**

---

## 🚀 Quick Start

### Generate Best Submission:
```bash
cd final_models
python final_ensemble_optimized.py
```

**Output:** `final_models/results/submission_final_ensemble_YYYYMMDD_HHMMSS.csv`

### Or Try Balanced Ensemble:
```bash
cd final_models  
python balanced_ensemble_search.py
```

---

## 📊 Available Submissions

All submission files are in `final_models/results/` and `final_models/results_balanced/`:

1. **Best Overall (0.5076):** XGBoost 80% + SVR 20%
2. **Balanced (0.5049):** XGBoost 60% + LightGBM 10% + CatBoost 10% + SVR 20%
3. **Most Balanced (0.5034):** XGBoost 50% + LightGBM 10% + CatBoost 20% + SVR 20%

---

## 🔑 Key Features

- ✅ KNN Imputation (k=5)
- ✅ Outlier Detection (IsolationForest 5%)
- ✅ ReliefF Feature Selection (k=250)
- ✅ Tuned Hyperparameters (Optuna optimization)
- ✅ Weighted Ensemble (Grid search optimized)
- ✅ Low Overfitting (17-19% gap)

---

## 📈 Model Evolution

| Version | CV R² | Notes |
|---------|-------|-------|
| Initial Baseline | 0.4881 | Tuned XGB/LGB/CatBoost |
| + SelectKBest | 0.5049 | f_regression k=200 |
| + ReliefF | 0.5050 | k=250 features |
| **+ Ensemble** | **0.5076** | **Optimized weights** |

---

## 🛠 Requirements

```bash
pip install -r requirements.txt
```

Main dependencies:
- scikit-learn 1.7.2
- xgboost
- lightgbm  
- catboost
- skrebate (ReliefF)
- optuna

---

## 📚 Documentation

- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - Complete directory guide
- **[documentation/FINAL_SUMMARY.md](documentation/FINAL_SUMMARY.md)** - Project summary
- **[documentation/TUNING_SUMMARY.md](documentation/TUNING_SUMMARY.md)** - Hyperparameter tuning
- **[documentation/FINAL_COMPARISON.md](documentation/FINAL_COMPARISON.md)** - Model comparisons

---

## 🗂 Folder Contents

### `final_models/` ⭐
**Production-ready models and best submissions**
- Ensemble scripts with optimized weights
- Final submission CSV files
- Best CV R² = 0.5076

### `experiments/`
**Feature selection and hyperparameter tuning**
- mRMR, ReliefF, SelectKBest comparison
- Optuna hyperparameter optimization
- SVR & ElasticNet tuning

### `analysis/`  
**Results, plots, and performance metrics**
- Feature selection comparisons
- Model performance visualizations
- CSV result files

### `utilities/`
**Reusable helper modules**
- Data loading (dataloader.py)
- Preprocessing pipeline
- Ensemble methods

### `archived/`
**Historical experiments (not used in final model)**
- Old scripts and approaches
- Previous output folders
- Early experiments

---

## 🎯 Recommended Usage

1. **For submission:** Use files in `final_models/results/`
2. **For experimentation:** Check `experiments/` folder
3. **For understanding:** Read `PROJECT_STRUCTURE.md`
4. **For historical context:** Browse `archived/`

---

**Last Updated:** November 1, 2025  
**Status:** ✅ Ready for Submission
