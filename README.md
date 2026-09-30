# Concrete Compressive Strength Prediction

Predicting the compressive strength of concrete (MPa) from its mix design and age, using machine learning guided by concrete technology principles.

**Best model:** Gradient Boosting with engineering-based features, **test RMSE 5.20 MPa, R² 0.896**.

**Try the live app:** https://concrete-strength-ml.streamlit.app

## Why this matters
Compressive strength is the main property engineers use to specify and accept concrete, but it is normally measured by crushing cubes or cylinders after 7 to 28 days of curing. A model that estimates strength from the mix design can help screen mix designs early, before waiting for lab results.

## Data

- **Source:** [Concrete Compressive Strength dataset](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength), UCI Machine Learning Repository, donated by Prof. I-Cheng Yeh
- **Dataset DOI:** [10.24432/C5PK67](https://doi.org/10.24432/C5PK67)
- **Original study:** Yeh, I-C. (1998). Modeling of strength of high-performance concrete using artificial neural networks. *Cement and Concrete Research*, 28(12).
- **Size:** 1,030 laboratory tests (1,005 after removing 25 duplicate rows), from 428 unique mix recipes
- **Inputs:** cement, blast furnace slag, fly ash, water, superplasticizer, coarse aggregate and fine aggregate (all in kg/m³), plus curing age (days)
- **Target:** compressive strength (MPa)
- **Licence:** CC BY 4.0
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
## Comparison with Abrams' law

To check whether ML adds value over a classic engineering formula, the model was compared with Abrams' law (1918), strength = A / B^(w/c), fitted on the same training data and evaluated on the same test set.

| Model | Test RMSE (MPa) | Test R² |
|---|---|---|
| Abrams (w/c only) | 13.39 | 0.313 |
| Abrams + log(age) | 11.59 | 0.485 |
| Gradient boosting (final model) | **5.20** | **0.896** |

Abrams' law captures the main trend, but the gradient boosting model cuts the error by about 55% compared with the best Abrams version, because it uses all the mix components and their interactions.

Final evaluation on the held-out test set: **RMSE 5.20 MPa, R² 0.896**. Domain features cut the linear model's error by about 30%, showing the value of engineering knowledge in feature design.

## Repository structure
```
concrete-strength-ml/
├── app/           Streamlit app (app.py)
├── data/          UCI dataset and its readme
├── models/        trained model (concrete_gb_model.joblib)
├── notebooks/     EDA and baseline modelling notebook
├── LICENSE
├── README.md
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