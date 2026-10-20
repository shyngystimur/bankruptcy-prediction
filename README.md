# Corporate Bankruptcy Prediction

Bachelor thesis project (KBTU, International School of Economics, University of London BSc Data Science and Business Analytics).

The goal is to predict whether a company will go bankrupt using its financial ratios, and to compare a classical statistical model with tree-based machine learning models.

## Data

[Company Bankruptcy Prediction](https://www.kaggle.com/competitions/company-bankruptcy-prediction/overview) competition on Kaggle, based on financial statements of Polish companies.

- 25,121 companies in the training set, 12,374 in the test set
- 64 financial ratios (profitability, liquidity, leverage, turnover) plus the forecasting period
- Strong class imbalance: only about 4.5% of companies are bankrupt

The data is not included in this repository. To run the notebook, download `train.csv` and `test.csv` from the competition page and place them next to the notebook.

## Approach

1. **Data checks:** missing values, descriptive statistics, comparison of bankrupt and non-bankrupt companies.
2. **Feature selection:** removal of near-constant features and one feature from each highly correlated pair (|r| ≥ 0.9), reducing 65 features to 41. Multicollinearity checked with VIF.
3. **Models:**
   - Logistic Regression (scaled features, balanced class weights)
   - Random Forest (400 trees, balanced class weights)
   - XGBoost (`scale_pos_weight` to handle class imbalance)
4. **Evaluation:** stratified 5-fold cross-validation plus a robustness check on three different random train/validation splits. Main metric is ROC-AUC, with precision, recall and F1 for the bankrupt class.
5. **Threshold analysis:** how the decision threshold changes the trade-off between catching bankrupt companies and raising false alarms.

## Results

Stratified 5-fold cross-validation (mean ± std):

| Model | ROC-AUC | F1 (bankrupt) | Precision | Recall |
|---|---|---|---|---|
| **XGBoost** | **0.931 ± 0.005** | **0.472 ± 0.021** | 0.356 | 0.702 |
| Random Forest | 0.905 ± 0.012 | 0.303 ± 0.043 | 0.830 | 0.186 |
| Logistic Regression | 0.673 ± 0.023 | 0.134 ± 0.011 | 0.076 | 0.572 |

![ROC curves](results/roc_curves_all_models.png)

**Key findings**

- XGBoost gives the best balance: it catches about 70% of bankrupt companies while keeping a ROC-AUC of 0.93.
- Random Forest is very precise (82% of its bankruptcy flags are correct) but misses most bankrupt companies, which makes it less useful for risk screening.
- Logistic Regression performs poorly here, mainly because the financial ratios contain extreme outliers and non-linear relationships.
- The most important predictors are profitability measures, such as gross profit over the last three years relative to total assets and operating profit relative to financial expenses.

![XGBoost feature importance](results/feature_importance_plot_xgboost.png)

**Choosing the threshold.** With the default threshold of 0.5, XGBoost has recall 0.68 and precision 0.36. Raising it to 0.7 increases precision to 0.63 and F1 to 0.58, at the cost of recall dropping to 0.54. A bank that wants to miss as few defaults as possible would keep a lower threshold; one that wants fewer false alarms would choose a higher one.

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.3 | 0.211 | 0.831 | 0.336 |
| 0.5 | 0.358 | 0.676 | 0.468 |
| 0.7 | 0.634 | 0.538 | 0.582 |

## Limitations

- Feature selection was done on the full training set before splitting, which may slightly overstate validation results. A stricter setup would repeat the selection inside each cross-validation fold.
- Logistic Regression could likely be improved with outlier treatment (winsorization or a robust scaler).
- Some Logistic Regression coefficients have counterintuitive signs (for example, retained earnings to total assets) because of remaining multicollinearity, so they should not be read as causal effects.

## Repository structure

```
bankruptcy_prediction.ipynb   full analysis
requirements.txt              Python libraries
results/                      key plots
```

## How to run

```bash
pip install -r requirements.txt
```

Place `train.csv` and `test.csv` next to the notebook, then open it in Jupyter and run all cells. All tables and plots are saved to the `results/` folder.

## Author

Timur Shyngys
Supervisor: Kairat Mynbaev
