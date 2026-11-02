# Medical Insurance Cost Prediction Using Machine Learning

## 1. Title Page
**Project Title:** Medical Insurance Cost Prediction Using Machine Learning (MEDINSIGHT AI)
**Submitted by:** Ashish Mahajan  | **Roll No.:** [your roll number]
**Course:** B.Tech Computer Science & Engineering, 7th Semester | **Subject:** Machine Learning
**Institution:** Sardar Beant Singh State University, Gurdaspur
**Guided by:** [teacher name] | **Academic Year:** 2026-27

## 2. Abstract
This project builds a web application that estimates annual medical insurance charges in US dollars from six attributes: age, sex, BMI, number of children, smoking status and region. A Linear Regression model, wrapped in a scikit-learn pipeline with one-hot encoding, is trained on the public Medical Cost Personal Dataset and evaluated on a held-out test set using MAE, RMSE and R². The saved pipeline is served through a Streamlit interface. The system is educational and is not an insurance quotation. Results: [INSERT actual MAE, RMSE, R² from `models/metrics.json`].

## 3. Introduction
Insurers use historical records to understand how costs vary between people. Machine Learning allows a computer to learn such relationships from data. This project applies supervised regression to a healthcare cost case study.

## 4. Problem Statement
Given a person's age, sex, BMI, number of children, smoking status and region, estimate the yearly medical insurance charges using patterns learned from historical data.

## 5. Objectives
- Understand and apply supervised Machine Learning and Linear Regression.
- Preprocess data and encode categorical features without data leakage.
- Evaluate the model honestly with MAE, RMSE and R².
- Save the trained pipeline and deploy it in an interactive local web application.
- Communicate limitations and ethical concerns clearly.

## 6. Scope
In scope: tabular prediction from six features, local web interface, data exploration and evaluation. Out of scope: real insurance quotations, medical diagnosis, non-US pricing, personal data storage, deep learning.

## 7. Existing System
Insurers traditionally use actuarial tables and manual underwriting rules. These are accurate in expert hands but are not easily accessible or transparent to students and the public, and a manual process is slow.

## 8. Proposed System
An automated pipeline that learns from historical data and gives an instant estimate through a web form, while showing the model's measured error and the underlying data.

## 9. Software and Hardware Requirements
**Software:** Windows 10/11, Python 3.10-3.12, VS Code, Streamlit, scikit-learn, Pandas, NumPy, Plotly, Joblib, a web browser.
**Hardware:** any laptop with at least 4 GB RAM (8 GB recommended); no GPU required; internet only for the one-time package and dataset download.

## 10. Dataset Description
Medical Cost Personal Datasets (`insurance.csv`), originally from Brett Lantz's *Machine Learning with R*, hosted on Kaggle.
| Column | Type | Meaning |
|---|---|---|
| age | numeric | Age of the primary beneficiary |
| sex | categorical | female / male |
| bmi | numeric | Body mass index |
| children | numeric | Number of dependents covered |
| smoker | categorical | yes / no |
| region | categorical | northeast, northwest, southeast, southwest |
| charges | numeric (target) | Yearly medical costs billed (USD) |
Number of records: [INSERT from the Data Explorer page]. Missing values: [INSERT from Data Explorer].

## 11. Mathematical Foundation of Linear Regression
The model predicts
ŷ = β₀ + β₁x₁ + β₂x₂ + … + βₚxₚ
where ŷ is the predicted charge, xⱼ are the (encoded) features, β₀ is the intercept and βⱼ are coefficients. Ordinary Least Squares chooses the coefficients that minimise the sum of squared errors:
J(β) = Σᵢ (yᵢ − ŷᵢ)²
Evaluation metrics:
- MAE = (1/n) Σ |yᵢ − ŷᵢ|
- RMSE = √[(1/n) Σ (yᵢ − ŷᵢ)²]
- R² = 1 − Σ(yᵢ − ŷᵢ)² / Σ(yᵢ − ȳ)²

## 12. System Architecture
Layers: (1) Data layer - `data/insurance.csv`; (2) ML layer - `src/model_utils.py` (validation, pipeline, training, evaluation); (3) Artifact layer - `models/insurance_model.joblib`, `models/metrics.json`; (4) Presentation layer - `app.py` with `assets/custom.css`. `train_model.py` runs the ML layer once; `app.py` only loads the saved artifacts.

## 13. Workflow Diagram
Historical Insurance Dataset → Data Loading → Data Validation and Preprocessing → Training and Testing Split → Linear Regression Training → Model Evaluation → Save Trained Pipeline → User Input → Prediction → Display Estimated Insurance Charges
*(Insert a drawn version of this diagram here if required by your department.)*

## 14. Algorithm
1. Load `insurance.csv` and validate columns, types, categories and missing values.
2. Separate features X (six columns) and target y (charges).
3. Split 80% training / 20% testing with random seed 42.
4. Build a pipeline: numeric columns pass through; categorical columns are one-hot encoded; then Linear Regression.
5. Fit the pipeline on training data only.
6. Predict on test data; compute MAE, RMSE, R².
7. Save the fitted pipeline (Joblib) and metrics (JSON).
8. In the app, validate user input, build a one-row table with the same columns, call `predict`, display the result.

## 15. Data Preprocessing
- Column names normalised (trimmed, lower case); required columns checked.
- Missing, non-numeric, infinite and out-of-range values rejected with clear messages.
- Categorical text normalised and checked against allowed values.
- OneHotEncoder converts sex, smoker and region into 0/1 columns; unknown categories are ignored safely.
- The encoder is fitted only on training data (prevents leakage).

## 16. Implementation Details
- `src/model_utils.py`: constants, `load_dataset`, `validate_dataframe`, `build_pipeline`, `train_and_evaluate`, `save_artifacts`, `load_model`, `load_metrics`, `predict_charge`.
- `train_model.py`: command-line training script.
- `app.py`: six-section Streamlit interface, cached loading, input validation, Plotly charts.
- Same constants are imported by both scripts, ensuring the feature schema matches.

## 17. Model Evaluation Methodology
The model is evaluated only on the 20% test set never seen during training. MAE gives average dollar error, RMSE emphasises large errors, and R² measures explained variance. R² is not classification accuracy. Metrics are computed at training time and stored; the app displays the stored values.

## 18. Results
| Metric | Value |
|---|---|
| MAE (USD) | [INSERT actual value] |
| RMSE (USD) | [INSERT actual value] |
| R² | [INSERT actual value] |
| Training rows / Test rows | [INSERT] / [INSERT] |
Interpretation: [Write 3-4 sentences in your own words using your real numbers, referring to the actual-vs-predicted chart.]

## 19. Screenshots
- Figure 1: Home page - [INSERT `screenshots/01_home.png`]
- Figure 2: Prediction form - [INSERT `02_prediction_form.png`]
- Figure 3: Prediction result - [INSERT `03_prediction_result.png`]
- Figure 4: Model performance - [INSERT `04_model_performance.png`]
- Figure 5: Data explorer - [INSERT `05_data_explorer.png`]
- Figure 6: Training terminal output - [INSERT `07_training_terminal.png`]

## 20. Limitations
- Linear Regression cannot capture complex non-linear effects or interactions on its own.
- Small dataset from a limited population; results may not generalise.
- Only six features; many real cost drivers are absent.
- Predictions in USD only; not applicable to other countries' premiums.
- The model can output unrealistic values for unusual inputs.

## 21. Ethical Considerations
Historical data may encode bias; learned associations are not causal; the tool must not be used alone for real insurance decisions or to discriminate; it does not diagnose illness; no personal data is collected or stored; users are told the output is an estimate.

## 22. Future Scope
Compare with Random Forest or Gradient Boosting; add cross-validation and residual analysis; use feature interactions (e.g. smoker × BMI); add prediction intervals; test on additional datasets; explainability tools.

## 23. Conclusion
The project delivers an end-to-end Machine Learning system: validated data, a leakage-safe pipeline, honest evaluation, persisted model and a professional web interface. It demonstrates how supervised regression can be applied to a healthcare cost problem while respecting its limits. [Add one sentence summarising your actual R².]

## 24. References
1. Lantz, B. *Machine Learning with R* (Packt Publishing) - original source of the insurance dataset.
2. Kaggle: "Medical Cost Personal Datasets" (insurance.csv).
3. Scikit-learn documentation: https://scikit-learn.org/stable/
4. Streamlit documentation: https://docs.streamlit.io/
5. Plotly Python documentation: https://plotly.com/python/
6. Pandas documentation: https://pandas.pydata.org/docs/
Add your course textbook/notes as per your university format.
