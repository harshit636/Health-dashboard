import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
try:
    import statsmodels.api as sm
    from statsmodels.stats.weightstats import ztest
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit.components.v1 as components

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# ---------------- PAGE ----------------
st.set_page_config(layout="wide", page_title="Rural Health Dashboard", page_icon="🏥")

# ---------------- 3D BACKGROUND & STYLING ----------------
st.markdown("""
<style>
/* Make the iframe containing the 3D background full screen and behind everything */
iframe {
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    width: 100vw !important;
    height: 100vh !important;
    z-index: -99 !important;
    border: none !important;
    pointer-events: none !important;
}

.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background-color: transparent !important;
}

/* Glassmorphism for conclusions and better typography */
.stAlert, div.stInfo, .css-1aumxhk, .stDataFrame {
    font-size: 1.05rem;
    background: rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(10px);
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.1);
    color: #c5c6c7;
}

h1, h2, h3 {
    color: #66fcf1 !important;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
}
</style>
""", unsafe_allow_html=True)

# ---------------- VANTA.JS 3D ANIMATION ----------------
components.html(
    """
    <div id="vanta-bg" style="width:100vw; height:100vh; position:absolute; top:0; left:0; overflow:hidden; margin:0; padding:0;"></div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r134/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/vanta@latest/dist/vanta.net.min.js"></script>
    <script>
    VANTA.NET({
      el: "#vanta-bg",
      mouseControls: true,
      touchControls: true,
      gyroControls: false,
      minHeight: 200.00,
      minWidth: 200.00,
      scale: 1.00,
      scaleMobile: 1.00,
      color: 0x66fcf1,
      backgroundColor: 0x0b0c10,
      points: 14.00,
      maxDistance: 22.00,
      spacing: 20.00
    })
    </script>
    """,
    height=0,
)

st.title("🏥 India Rural Health Infrastructure Analysis")
st.markdown("---")

# ---------------- LOAD DATA ----------------
try:
    df = pd.read_csv("dataset.csv")
    st.header("📊 Dataset Overview")
    st.dataframe(df.head())
    
    numeric_cols_count = len(df.select_dtypes(include=np.number).columns)
    st.success(f"📝 **Conclusion:** The dataset contains **{df.shape[0]} rows** and **{df.shape[1]} columns**. We have {numeric_cols_count} numeric features available for quantitative modeling.")
except FileNotFoundError:
    st.error("⚠️ Dataset 'dataset.csv' not found. Please ensure the file is in the same directory.")
    st.stop()

# ---------------- CLEAN DATA ----------------
numeric_df = df.select_dtypes(include=np.number)
if numeric_df.empty:
    st.error("No numeric columns found in the dataset for analysis.")
    st.stop()

# Fill missing values
numeric_df = numeric_df.fillna(numeric_df.mean())

# ---------------- MISSING VALUES ----------------
st.header("🧹 Missing Value Analysis")
col1, col2 = st.columns(2)

missing = df.isnull().sum()

with col1:
    st.write(missing)
    st.bar_chart(missing)

with col2:
    plt.figure(figsize=(8,5))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
    st.pyplot(plt)

most_missing_val = missing.max()
if most_missing_val > 0:
    most_missing_col = missing.idxmax()
    st.success(f"📝 **Conclusion:** The feature with the highest missing values is **{most_missing_col}** with **{most_missing_val}** missing entries. These gaps have been filled with the column mean to prevent biased machine learning models.")
else:
    st.success("📝 **Conclusion:** The dataset is clean with **0 missing values** across all columns. No data imputation was needed!")


# ---------------- DISTRIBUTION ANALYSIS ----------------
st.header("📈 Distribution Analysis")
cols = st.columns(3)
for i, col in enumerate(numeric_df.columns):
    fig = px.histogram(numeric_df, x=col, marginal="box", color_discrete_sequence=['#45a29e'])
    cols[i % 3].plotly_chart(fig, use_container_width=True)

skewness = numeric_df.skew().abs().idxmax()
st.success(f"📝 **Conclusion:** The histograms reveal the spread of the data. For instance, **{skewness}** shows the highest skewness, indicating a long tail in its distribution which might require transformation (e.g., log scale) for certain predictive models.")

# ---------------- CORRELATION ----------------
st.header("🔗 Correlation Heatmap")
corr = numeric_df.corr()
plt.figure(figsize=(14,10))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
st.pyplot(plt)

corr_unstacked = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
if not corr_unstacked.empty:
    highest_corr_pair = corr_unstacked.idxmax()
    highest_corr_val = corr_unstacked.max()
    st.success(f"📝 **Conclusion:** The strongest positive correlation is between **{highest_corr_pair[0]}** and **{highest_corr_pair[1]}** (r = {highest_corr_val:.2f}). Highly correlated predictor pairs might indicate redundant information and should be noted for feature selection.")
else:
    st.success("📝 **Conclusion:** The correlation heatmap identifies linear relationships between features.")


# ---------------- NEW: 3D SCATTER PLOT ----------------
st.header("🌌 3D Feature Relationships")
if len(numeric_df.columns) >= 3:
    fig = px.scatter_3d(numeric_df, x=numeric_df.columns[0], y=numeric_df.columns[1], z=numeric_df.columns[2], 
                        color=numeric_df.columns[3] if len(numeric_df.columns) > 3 else None,
                        opacity=0.7)
    fig.update_layout(scene=dict(xaxis_title=numeric_df.columns[0], yaxis_title=numeric_df.columns[1], zaxis_title=numeric_df.columns[2]))
    st.plotly_chart(fig, use_container_width=True)
    st.success(f"📝 **Conclusion:** This 3D view plots **{numeric_df.columns[0]}**, **{numeric_df.columns[1]}**, and **{numeric_df.columns[2]}**. It helps visualize multidimensional clusters and any outliers existing simultaneously across these three key features.")


# ---------------- NEW: PAIRPLOT / SCATTER MATRIX ----------------
st.header("🌐 Scatter Matrix")
cols_to_plot = numeric_df.columns[:4]
fig = px.scatter_matrix(numeric_df, dimensions=cols_to_plot)
fig.update_layout(height=800)
st.plotly_chart(fig, use_container_width=True)
st.success(f"📝 **Conclusion:** The scatter matrix cross-examines the top features: **{', '.join(cols_to_plot)}**. By viewing these pairs simultaneously, we can rapidly spot linear trends or complex clusters across the primary variables.")


# ---------------- TOP STATES ----------------
if "State" in df.columns:
    st.header("🏆 Top States")
    col = numeric_df.columns[0]
    top_states = df.groupby("State")[col].mean().nlargest(15).reset_index()
    fig = px.bar(top_states, x="State", y=col, color=col, color_continuous_scale="Tealgrn")
    st.plotly_chart(fig, use_container_width=True)
    
    top_state_name = top_states.iloc[0]['State']
    top_state_val = top_states.iloc[0][col]
    st.success(f"📝 **Conclusion:** **{top_state_name}** leads the ranking for '{col}' with an average value of **{top_state_val:.2f}**. This highlights the region as a top performer or highest-need area depending on the metric's nature.")

# ---------------- YEAR TREND ----------------
if "Year" in df.columns:
    st.header("📅 Year-over-Year Trend")
    yearly = df.groupby("Year")[numeric_df.columns].mean().reset_index()
    fig = px.area(yearly, x="Year", y=numeric_df.columns[0], color_discrete_sequence=['#66fcf1'])
    st.plotly_chart(fig, use_container_width=True)
    
    if len(yearly) > 1:
        start_year_val = yearly.iloc[0][numeric_df.columns[0]]
        end_year_val = yearly.iloc[-1][numeric_df.columns[0]]
        trend = "increased" if end_year_val > start_year_val else "decreased"
        st.success(f"📝 **Conclusion:** Overall, **{numeric_df.columns[0]}** has **{trend}** from {start_year_val:.2f} to {end_year_val:.2f} over the recorded timeframe, indicating the long-term trajectory of this metric.")
    else:
        st.success(f"📝 **Conclusion:** Data for {numeric_df.columns[0]} is recorded for a single year.")


# ---------------- OUTLIERS ----------------
st.header("📦 Outlier Detection")
cols = st.columns(3)
outlier_counts = {}

for i, col in enumerate(numeric_df.columns):
    fig = px.box(numeric_df, y=col, color_discrete_sequence=['#EF553B'])
    cols[i % 3].plotly_chart(fig, use_container_width=True)
    
    # Calculate outliers
    Q1 = numeric_df[col].quantile(0.25)
    Q3 = numeric_df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((numeric_df[col] < (Q1 - 1.5 * IQR)) | (numeric_df[col] > (Q3 + 1.5 * IQR))).sum()
    outlier_counts[col] = outliers

most_outliers_col = max(outlier_counts, key=outlier_counts.get)
st.success(f"📝 **Conclusion:** **{most_outliers_col}** contains the most extreme values (**{outlier_counts[most_outliers_col]}** outliers). Investigating these specific data points is crucial to understand unusual healthcare scenarios or anomalies.")


# ---------------- STATISTICAL ANALYSIS ----------------
st.header("🧮 Statistical Analysis")

if len(numeric_df.columns) >= 2:
    col1, col2 = numeric_df.columns[0], numeric_df.columns[1]
    
    st.subheader("Hypothesis Testing & Distributions")
    tab1, tab2 = st.tabs(["T-Test", "Z-Test"])
    
    with tab1:
        st.markdown(f"**Independent T-Test** between `{col1}` and `{col2}`")
        t_stat, p_val_t = stats.ttest_ind(numeric_df[col1].dropna(), numeric_df[col2].dropna())
        st.write(f"T-statistic: `{t_stat:.4f}` | P-value: `{p_val_t:.4e}`")
        if p_val_t < 0.05:
            st.success(f"📝 **Conclusion:** P-value < 0.05. We reject the null hypothesis. There is a significant difference between the means of {col1} and {col2}.")
        else:
            st.info(f"📝 **Conclusion:** P-value >= 0.05. We fail to reject the null hypothesis. No significant difference between the means of {col1} and {col2}.")

    with tab2:
        st.markdown(f"**Z-Test** between `{col1}` and `{col2}`")
        if HAS_STATSMODELS:
            z_stat, p_val_z = ztest(numeric_df[col1].dropna(), numeric_df[col2].dropna())
            st.write(f"Z-statistic: `{z_stat:.4f}` | P-value: `{p_val_z:.4e}`")
            if p_val_z < 0.05:
                st.success("📝 **Conclusion:** P-value < 0.05. Significant difference found using Z-test.")
            else:
                st.info("📝 **Conclusion:** P-value >= 0.05. No significant difference found using Z-test.")
        else:
            st.warning("`statsmodels` library is required for Z-Test. Please install it (`pip install statsmodels`).")

# ---------------- MACHINE LEARNING ----------------
st.header("🤖 Machine Learning Insights")

if len(numeric_df.columns) > 1:
    X = numeric_df.iloc[:, :-1]
    y = numeric_df.iloc[:, -1]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # -------- REGRESSION --------
    st.subheader("Simple Linear Regression Performance")
    
    feature_col = X.columns[0]
    st.markdown(f"**Predicting `{y.name}` using `{feature_col}`**")
    
    X_train_simple = X_train[[feature_col]]
    X_test_simple = X_test[[feature_col]]
    
    model = LinearRegression()
    model.fit(X_train_simple, y_train)
    pred = model.predict(X_test_simple)
    
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    
    metrics = pd.DataFrame({"Metric": ["MAE", "RMSE", "R2 Score"], "Value": [mae, rmse, r2]})
    fig = px.bar(metrics, x="Metric", y="Value", color="Metric", text="Value")
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)
    st.success(f"📝 **Conclusion:** The Simple Linear Regression model achieved an R² score of **{r2:.3f}**. This means the model explains {r2*100:.1f}% of the variance in the target variable using the single feature '{feature_col}', with an average error (MAE) of **{mae:.2f}**.")

    # -------- REGRESSION FIT PLOT --------
    st.subheader(f"Regression Line: {feature_col} vs {y.name}")
    fig = px.scatter(x=X_test_simple[feature_col], y=y_test, trendline="ols", trendline_color_override="red")
    fig.update_layout(xaxis_title=feature_col, yaxis_title=y.name)
    st.plotly_chart(fig, use_container_width=True)
    st.success(f"📝 **Conclusion:** The scatter plot visualizes the Simple Linear Regression fit. The spread around the trendline (RMSE = **{rmse:.2f}**) visualizes the predictive accuracy and variance of this model.")


    # -------- FEATURE IMPORTANCE --------
    st.subheader("Feature Importance (Random Forest)")
    rf = RandomForestRegressor(random_state=42)
    rf.fit(X, y)
    importance = pd.DataFrame({"Feature": X.columns, "Importance": rf.feature_importances_}).sort_values(by="Importance", ascending=False)
    fig = px.bar(importance, x="Feature", y="Importance", color="Importance", color_continuous_scale="Plasma")
    st.plotly_chart(fig, use_container_width=True)
    
    top_feature = importance.iloc[0]['Feature']
    top_importance = importance.iloc[0]['Importance']
    st.success(f"📝 **Conclusion:** The Random Forest model identifies **{top_feature}** as the most critical driver (Importance: **{top_importance:.2f}**). This variable should be the primary focus for any strategic interventions.")
else:
    st.warning("Not enough numeric columns for Machine Learning.")