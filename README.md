# House Price Prediction: Ames Housing (Regression)

> A Python and scikit-learn regression pipeline that predicts house sale prices in Ames, Iowa from numerical property features only.

## Business Problem
How well can a simple, well-prepared regression model predict a home's sale price from numerical features such as square footage, room counts, lot size and year built? Buyers, sellers and lenders all need fair price estimates. This project tests how far careful preprocessing and feature engineering can take a linear model.

## Dataset
- **Source:** [Kaggle: House Prices – Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) (Ames Housing Dataset)
- **Files:** `train.csv`, `test.csv`. These are not included in this repo; download them from Kaggle.
- **Target:** `SalePrice` (USD), modelled as `log(1 + SalePrice)`
- **Features used:** numerical columns only (`int64` / `float64`), for example OverallQual, GrLivArea, GarageCars, GarageArea, TotalBsmtSF, YearBuilt, FullBath, TotRmsAbvGrd, LotArea

## Tools Used
Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · Google Colab

## Approach
1. **Load and filter:** read train/test data and keep numerical features only.
2. **EDA:** rank features by correlation with SalePrice, plot a heatmap of the top features, draw scatterplots of the top 3 against price, and plot the SalePrice distribution and skewness.
3. **Target transform:** SalePrice is right-skewed, so `np.log1p` is applied.
4. **Outliers:** checked with boxplots and kept as genuine sales.
5. **Missing values:** zero-fill for structural absences (GarageYrBlt, GarageArea, GarageCars, MasVnrArea) and median imputation for everything else (for example LotFrontage).
6. **Feature engineering:**

   | Feature | Formula |
   |---|---|
   | TotalSF | 1stFlrSF + 2ndFlrSF + TotalBsmtSF |
   | HouseAge | YrSold − YearBuilt |
   | YearsSinceRemodel | YrSold − YearRemodAdd |
   | TotalBathrooms | FullBath + 0.5×HalfBath + BsmtFullBath + 0.5×BsmtHalfBath |

7. **Multicollinearity check:** flagged feature pairs with |r| > 0.80 and reviewed them; some were kept.

## Model & Evaluation
- **Models:** `Pipeline(StandardScaler → LinearRegression)`, with `Pipeline(StandardScaler → Ridge(alpha=10))` as a comparison
- **Validation:** 5-fold `KFold` (shuffle=True, random_state=42)
- **Metric:** RMSE on log(1 + SalePrice)
- **Reported result (project report):** Linear Regression reached a cross-validated **RMSE < 0.16** on log sale price.
- Re-running the script prints the per-fold and average RMSE for both models and states which one performed better.

## Key Findings
- The strongest correlates of SalePrice, in order, are **OverallQual > GrLivArea > GarageCars > GarageArea > TotalBsmtSF**.
- GrLivArea has a strong positive linear relationship with price. GarageCars and TotalBsmtSF are also positive, with more variance.
- SalePrice is right-skewed: many moderately priced homes and a few expensive ones. A log transform was needed.
- The model's biggest errors are on **luxury and unusual homes**, whose distinctive traits numerical features don't capture.

## Recommendations / Next Steps
- Add categorical features such as neighbourhood, condition and quality ratings. This is the obvious way to improve on luxury and unusual homes.
- Tune Ridge and Lasso regularization, and try Random Forest or XGBoost.
- Engineer more features (for example quality × area interactions).

## Repository Structure
```
HOUSE-PREDICTION-ANALYSIS-/
├── assignment_week_8_.py        # Full pipeline (exported from Google Colab)
├── House Price Prediction.pdf   # Project report
└── README.md
```

## How to Run
1. Download `train.csv` and `test.csv` from the [Kaggle competition page](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data).
2. Open `assignment_week_8_.py` in **Google Colab** (it uses `google.colab.files.upload()`), upload both CSVs when prompted, and run all cells.
3. To run locally, remove the `google.colab` upload lines, put the CSVs next to the script, then run:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   python assignment_week_8_.py
   ```

## Author
**Anointed Ubongabasi Akai**, Data Analyst
[LinkedIn](https://www.linkedin.com/in/akaianointed) · [GitHub](https://github.com/OfficialAKAIANOINTED)
