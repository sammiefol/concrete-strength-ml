# Concrete Compressive Strength Prediction

Predicting the compressive strength of concrete (MPa) from its mix design and age, using machine learning guided by concrete technology principles.

**Best model:** Gradient Boosting with engineering-based features, **test RMSE 5.20 MPa, R² 0.896**.

## Why this matters
Compressive strength is the main property engineers use to specify and accept concrete, but it is normally measured by crushing cubes or cylinders after 7 to 28 days of curing. A model that estimates strength from the mix design can help screen mix designs early, before waiting for lab results.

## Data
[UCI Concrete Compressive Strength dataset](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength) (I-Cheng Yeh): 1,030 laboratory tests with 8 inputs (cement, blast furnace slag, fly ash, water, superplasticizer, coarse aggregate, fine aggregate, age) and 1 target (compressive strength). After removing 25 duplicate rows, 1,005 tests remain.

## Approach
- **Engineering-informed EDA:** the water–cement ratio correlates more strongly with strength (Spearman −0.50) than water alone (−0.28), consistent with Abrams' law.
- **Leakage-safe validation:** the 1,005 tests come from only 428 unique mix recipes, many tested at several ages. Data is split by mix recipe, not by row, so no recipe appears in both training and test sets.
- **Baseline model ladder:** from a mean-predicting dummy model to gradient boosting, each with and without domain features (water–cement ratio, log of age).

## Results

| Model | Features | CV RMSE (MPa) |
|---|---|---|
| Dummy (predicts mean) | – | 16.34 |
| Linear Regression | raw | 10.70 |
| Linear Regression | raw + domain | 7.46 |
| Random Forest | raw | 6.53 |
| Random Forest | raw + domain | 6.28 |
| Gradient Boosting | raw | 6.32 |
| **Gradient Boosting** | **raw + domain** | **6.11** |

Final evaluation on the held-out test set: **RMSE 5.20 MPa, R² 0.896**. Domain features cut the linear model's error by about 30%, showing the value of engineering knowledge in feature design.

## Repository structure
```
concrete-strength-ml/
├── data/          UCI dataset and its readme
├── models/        trained model (concrete_gb_model.joblib)
├── notebooks/     EDA and baseline modelling notebook
├── app/           Streamlit app (coming soon)
└── requirements.txt
```

## How to run
```
git clone https://github.com/sammiefol/concrete-strength-ml.git
cd concrete-strength-ml
pip install -r requirements.txt
```
Then open `notebooks/01_eda_baseline.ipynb` in Jupyter or VS Code.

## Next steps
- Tune the gradient boosting model
- Compare LightGBM, XGBoost and CatBoost
- Build a Streamlit app that predicts strength from a user-entered mix design

## Acknowledgements
Dataset by I-Cheng Yeh, made available through the UCI Machine Learning Repository under a CC BY 4.0 licence. Original study: Yeh, I-C. (1998), "Modeling of strength of high-performance concrete using artificial neural networks", *Cement and Concrete Research*, 28(12), 1797–1808.

## Author
**Samuel Folorunso**, Civil and Environmental Engineering, University of Lagos