# House Price Prediction (Ames Housing)

**Question:** How well can a simple, well-prepared regression model predict house sale prices using only numerical property features?

**Data:** [Kaggle House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) (Ames, Iowa). The training set has 1,460 homes. The target is `SalePrice` in US dollars.

## Approach

1. **Feature selection:** kept numerical columns only (square footage, room counts, lot size, year built and similar).
2. **Exploration:** the strongest correlates of price were OverallQual, GrLivArea, GarageCars, GarageArea and TotalBsmtSF. `SalePrice` was right-skewed, so the model predicts `log(1 + SalePrice)`.
3. **Cleaning:** median imputation for continuous gaps (for example LotFrontage). Zero for structural absences (no garage, no masonry veneer). Outliers were kept as genuine sales.
4. **Feature engineering:**

   | Feature | Formula |
   |---|---|
   | TotalSF | 1stFlrSF + 2ndFlrSF + TotalBsmtSF |
   | HouseAge | YrSold − YearBuilt |
   | YearsSinceRemodel | YrSold − YearRemodAdd |
   | TotalBathrooms | FullBath + 0.5 × HalfBath + BsmtFullBath + 0.5 × BsmtHalfBath |

5. **Modelling:** `StandardScaler` → Linear Regression, compared with Ridge (alpha = 10). Both were scored by RMSE on log price with 5-fold cross-validation.

## Results

The project write-up (`House Price Prediction.pdf`) reports a cross-validated **RMSE below 0.16 on log sale price**.

## Key findings

- Build quality and living area are the strongest price drivers. Garage and basement size also matter.
- The model's largest errors are on luxury and unusual homes, which numerical features alone do not describe well.

## Limitations and next steps

- Categorical features (neighbourhood, condition and so on) were left out on purpose. Adding them is the obvious next step.
- Try Lasso, tuned Ridge, Random Forest or XGBoost.

## Files

- `assignment_week_8_.py`: full pipeline, exported from Google Colab (to be renamed `house_price_prediction.py` or uploaded as a notebook)
- `House Price Prediction.pdf`: short project report

## How to run

Download `train.csv` and `test.csv` from Kaggle, open the script or notebook in Google Colab, upload both files when prompted, and run all cells.

**Tools:** Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

**Author:** Anointed Akai · [LinkedIn](https://www.linkedin.com/in/akaianointed)
