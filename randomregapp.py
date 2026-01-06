import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Housing Insights & Predictor",
    layout="wide"
)

st.title("🏠 Housing Price Predictor with Insights")

# -----------------------------
# Load Data & Train Model
# -----------------------------
@st.cache_resource
def train_model():
    try:
        df = pd.read_csv(
            "https://raw.githubusercontent.com/benanush/random-forest-regression/main/randomregc.csv"
        )
    except Exception as e:
        st.error(f"❌ Failed to load dataset: {e}")
        st.stop()

    # Target
    X = df.drop('Price_USD', axis=1)
    y = df['Price_USD']

    cat_features = X.select_dtypes(include=['object']).columns.tolist()
    num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ]
    )

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=100,
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)

    test_preds = model.predict(X_test)

    return model, df, cat_features, num_features, y_test, test_preds

model, df, cat_cols, num_cols, y_test, test_preds = train_model()

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("🛠️ Property Details")
user_inputs = {}

for col in cat_cols:
    user_inputs[col] = st.sidebar.selectbox(col, df[col].unique())

for col in num_cols:
    user_inputs[col] = st.sidebar.slider(
        col,
        float(df[col].min()),
        float(df[col].max()),
        float(df[col].mean())
    )

# -----------------------------
# Main Layout
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔮 Make a Prediction")
    if st.button("Calculate Predicted Price"):
        input_df = pd.DataFrame([user_inputs])
        prediction = model.predict(input_df)[0]
        st.success(f"Estimated Price: **${prediction:,.2f}**")
        st.balloons()

with col2:
    st.subheader("📈 Model Performance")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_test, test_preds, alpha=0.5)
    ax.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        '--r'
    )
    ax.set_xlabel("Actual Price ($)")
    ax.set_ylabel("Predicted Price ($)")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig)

st.divider()

# -----------------------------
# Feature Importance
# -----------------------------
st.subheader("📊 What drives the price?")
col3, col4 = st.columns([2, 1])

with col3:
    ohe = model.named_steps['preprocessor'].transformers_[1][1]
    ohe_cols = ohe.get_feature_names_out(cat_cols).tolist()

    all_features = num_cols + ohe_cols
    importances = model.named_steps['regressor'].feature_importances_

    feat_importances = (
        pd.Series(importances, index=all_features)
        .sort_values(ascending=False)
        .head(10)
    )

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    feat_importances.plot(kind='barh', ax=ax2)
    ax2.set_title("Top 10 Factors Influencing House Price")
    ax2.invert_yaxis()
    st.pyplot(fig2)

with col4:
    st.info("""
    **Understanding the Chart**
    
    The bar chart shows the most influential features used by the Random Forest model
    to estimate house prices.
    
    Higher bars indicate stronger impact on pricing decisions.
    """)
