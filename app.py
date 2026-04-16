import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ---------------- PAGE ----------------
st.set_page_config(layout="wide", page_title="Health Dashboard")

st.title("🏥 Rural Health Analytics Dashboard")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("dataset.csv")

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(include="object").columns.tolist()

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Controls")

selected_num = st.sidebar.selectbox("Select Numeric Column", numeric_cols)
selected_cat = st.sidebar.selectbox("Select Category", cat_cols)

# ---------------- METRICS ----------------
col1, col2, col3 = st.columns(3)

col1.metric("Total Records", len(df))
col2.metric("Mean Value", round(df[selected_num].mean(), 2))
col3.metric("Max Value", round(df[selected_num].max(), 2))

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "📈 Visualization",
    "🔗 Correlation",
    "🤖 ML Prediction"
])

# ================= TAB 1 =================
with tab1:
    st.subheader("Dataset Overview")
    st.dataframe(df.head())

    st.subheader("Summary")
    st.write(df.describe())

# ================= TAB 2 =================
with tab2:
    st.subheader("Visualizations")

    colA, colB = st.columns(2)

    with colA:
        st.plotly_chart(px.histogram(df, x=selected_num), use_container_width=True)

        st.plotly_chart(px.box(df, y=selected_num), use_container_width=True)

    with colB:
        st.plotly_chart(px.bar(df.groupby(selected_cat)[selected_num].mean().reset_index(),
                               x=selected_cat, y=selected_num),
                        use_container_width=True)

        st.plotly_chart(px.scatter(df, x=selected_num, y=numeric_cols[1]),
                        use_container_width=True)

# ================= TAB 3 =================
with tab3:
    st.subheader("Correlation Heatmap")

    corr = df[numeric_cols].corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm")

    st.pyplot(plt)

# ================= TAB 4 =================
with tab4:
    st.subheader("Machine Learning")

    x_col = st.selectbox("Feature", numeric_cols)
    y_col = st.selectbox("Target", numeric_cols)

    if x_col != y_col:
        X = df[[x_col]].fillna(df[x_col].mean())
        y = df[y_col].fillna(df[y_col].mean())

        model = LinearRegression()
        model.fit(X, y)

        pred = model.predict(X)

        fig = px.scatter(x=X[x_col], y=y)
        fig.add_scatter(x=X[x_col], y=pred, mode='lines')

        st.plotly_chart(fig, use_container_width=True)

        st.write("R² Score:", r2_score(y, pred))