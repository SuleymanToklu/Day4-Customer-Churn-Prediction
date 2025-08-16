# 👋 Day 4: Telco Customer Churn Prediction

This is the fourth project of my #30DaysOfAI challenge. The goal is to build a classification model that predicts whether a customer will churn (leave the company).

### ✨ Key Concepts
* **Classification with XGBoost:** A classic and powerful model for a common business problem.
* **Handling Imbalanced Data:** Using the `scale_pos_weight` parameter to give more importance to the minority class (customers who churn).
* **End-to-End Application:** From a raw CSV to a fully interactive web application where users can simulate different customer profiles and see the churn probability.

### 💻 Tech Stack
- Python, Pandas, Scikit-learn, XGBoost, Streamlit

### 🚀 How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Train the model: `python train_model.py`
3. Run the app: `streamlit run app.py`