# Presentation Outline - MEDINSIGHT AI (10 slides)
Replace bracketed items with real values from your running app.

1. **Title** - Medical Insurance Cost Prediction Using Machine Learning; your name, roll no., university (Sardar Beant Singh State University, Gurdaspur), subject teacher.
2. **Introduction & Problem** - Insurance charges vary by person; goal: estimate yearly charges from historical data; educational only.
3. **Objectives** - Apply supervised regression; preprocess and encode data; evaluate honestly; deploy a web interface.
4. **Dataset & Features** - Medical Cost Personal Dataset; [rows] records; inputs age, sex, BMI, children, smoker, region; target charges (USD).
5. **Algorithm** - Linear Regression: charges ~ intercept + weighted sum of features; OneHotEncoder for categories; scikit-learn Pipeline.
6. **Architecture & Workflow** - Dataset -> Load -> Validate/Preprocess -> 80/20 split -> Train -> Evaluate -> Save (Joblib) -> User input -> Predict -> Display.
7. **Website** - Streamlit pages: Home, Prediction, Model Performance, Data Explorer, About, How It Works; add 2-3 screenshots.
8. **Evaluation** - MAE [x], RMSE [x], R2 [x] from the test set; actual-vs-predicted chart; R2 is not accuracy.
9. **Limitations & Future Scope** - Simple linear model, small US dataset, bias, no causation; future: Random Forest/Gradient Boosting comparison, cross-validation, more features.
10. **Conclusion** - Working end-to-end ML app; honest metrics; thank you / questions.

# Viva Questions and Answers (speak in your own words)
1. **What is Machine Learning?** Teaching computers to learn patterns from data instead of writing fixed rules. Here the computer learns how age, BMI, smoking etc. relate to charges.
2. **What is supervised learning?** Learning from examples that already have the correct answer. Each insurance record includes the real charge, so the model learns from it.
3. **What is regression?** Predicting a number (like a dollar amount), as opposed to classification, which predicts a category.
4. **What is Linear Regression?** It finds the best straight-line-style formula: charges = base value + (weight x age) + (weight x BMI) + ... The weights are learned from data.
5. **What is the target variable?** The thing we predict: `charges`.
6. **What are features?** The inputs used to predict: age, sex, BMI, children, smoker, region.
7. **Why OneHotEncoder?** Models need numbers. It turns text like "northeast" into separate 0/1 columns, so no fake order (1 < 2 < 3) is introduced.
8. **Why split training and testing data?** Train on 80%, test on unseen 20% to check the model works on new data, not just memorised data.
9. **What is data leakage?** When information from test data sneaks into training (e.g. fitting the encoder on all data). We avoid it by fitting only on training data inside a Pipeline.
10. **What is MAE?** Mean Absolute Error: average dollar difference between predicted and actual charges. Easy to read.
11. **What is RMSE?** Root Mean Squared Error: like MAE, but big mistakes are penalised more. Also in dollars.
12. **What is R2?** How much of the variation in charges the model explains. 1 is perfect, 0 is no better than the average. It is not "accuracy".
13. **Why save the trained model?** So we train once and reuse it instantly for every prediction; Joblib stores the fitted pipeline in a file.
14. **How does Streamlit work?** You write Python; Streamlit turns it into a web page and re-runs the script whenever a user interacts with a widget.
15. **Limitations?** Linear assumptions, limited data from one population, possible bias, correlation not causation, not a real quote, and it should not decide real insurance cases.
