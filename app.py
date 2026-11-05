
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AeroSense | AQI Intelligence",
    page_icon="🌿",
    layout="wide"
)

# ==========================================
# 2. CUSTOM CSS
# ==========================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #f4f8f7;
    color: #172d29;
}

[data-testid="stSidebar"] {
    background: #102d29;
}

[data-testid="stSidebar"] * {
    color: white !important;
}

.main-title {
    font-size: 35px;
    font-weight: 800;
    color: #163d35;
}

.subtitle {
    color: #71847e;
    font-size: 15px;
}

.metric-card {
    background: white;
    padding: 22px;
    border-radius: 15px;
    border: 1px solid #e4eeea;
    min-height: 125px;
}

.metric-label {
    color: #81918b;
    font-size: 13px;
    font-weight: 600;
}

.metric-value {
    color: #173f35;
    font-size: 29px;
    font-weight: 800;
    margin-top: 8px;
}

div.stButton > button {
    background: #13866b;
    color: white;
    border: none;
    border-radius: 10px;
    height: 45px;
    font-weight: 700;
}

div.stButton > button:hover {
    background: #0d6d56;
    color: white;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. CHECK FILES
# ==========================================

if not os.path.exists("models/aqi_model.pkl"):
    st.error("Model not found. Run python train_model.py first.")
    st.stop()

if not os.path.exists("models/model_metrics.pkl"):
    st.error("Model metrics not found. Train the model again.")
    st.stop()

if not os.path.exists("data/city_day.csv"):
    st.error("Dataset not found inside the data folder.")
    st.stop()


# ==========================================
# 4. LOAD MODEL AND DATASET
# ==========================================

@st.cache_resource
def load_model():
    return joblib.load("models/aqi_model.pkl")


@st.cache_data
def load_data():
    return pd.read_csv("data/city_day.csv")


@st.cache_data
def load_metrics():
    return joblib.load("models/model_metrics.pkl")


bundle = load_model()
df = load_data()
metrics = load_metrics()

model = bundle["model"]
features = bundle["features"]
medians = bundle["medians"]
best_name = bundle["best_name"]


# ==========================================
# 5. DATA CLEANING
# ==========================================

df["Date"] = pd.to_datetime(
    df["Date"], errors="coerce"
)

# Convert AQI into numeric format
df["AQI"] = pd.to_numeric(
    df["AQI"], errors="coerce"
)

# Convert pollutant columns into numeric format
for feature in features:
    if feature in df.columns:
        df[feature] = pd.to_numeric(
            df[feature], errors="coerce"
        )

# Remove missing and invalid AQI records
df = df.dropna(subset=["AQI"])

# Keep only valid Indian AQI range
df = df[df["AQI"].between(0, 500)].copy()

if df.empty:
    st.error("No valid AQI records found in the dataset.")
    st.stop()


# ==========================================
# 6. AQI CATEGORY FUNCTION
# ==========================================

def category(aqi):

    if aqi <= 50:
        return "Good", "🟢"

    elif aqi <= 100:
        return "Satisfactory", "🟡"

    elif aqi <= 200:
        return "Moderate", "🟠"

    elif aqi <= 300:
        return "Poor", "🔴"

    elif aqi <= 400:
        return "Very Poor", "🟣"

    else:
        return "Severe", "⚫"


# ==========================================
# 7. HEALTH RECOMMENDATIONS
# ==========================================

advice = {
    "Good": "Air quality is satisfactory. Enjoy normal outdoor activities.",

    "Satisfactory": "Sensitive individuals should monitor their symptoms.",

    "Moderate": "Sensitive people should reduce prolonged outdoor exertion.",

    "Poor": "Reduce prolonged outdoor activities and take precautions.",

    "Very Poor": "Avoid strenuous outdoor activities and limit exposure.",

    "Severe": "Avoid outdoor physical activity and follow public health advice."
}


# ==========================================
# 8. SIDEBAR NAVIGATION
# ==========================================

with st.sidebar:

    st.markdown("# 🌿 AeroSense")

    st.caption("ENVIRONMENTAL INTELLIGENCE")

    st.divider()

    page = st.radio(
        "MAIN MENU",
        [
            "Overview",
            "Predict AQI",
            "City Explorer",
            "Pollution Insights",
            "Model Lab"
        ]
    )

    st.divider()

    st.markdown("### About")

    st.caption(
        "AI-powered air quality analysis "
        "using environmental data."
    )


# ==========================================
# 9. MAIN HEADER
# ==========================================

st.markdown(
    '<div class="main-title">Environmental Intelligence</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Monitor. Analyze. Predict. Protect our environment.'
    '</div>',
    unsafe_allow_html=True
)

st.write("")


# ==========================================
# PAGE 1: OVERVIEW DASHBOARD
# ==========================================

if page == "Overview":

    st.markdown("### Overview Dashboard")

    valid = df["AQI"]

    # Dashboard metrics
    metrics_data = [
        ("🌍", "AVERAGE AQI", f"{valid.mean():.1f}"),

        ("🏙️", "CITIES MONITORED",
         str(df["City"].nunique())),

        ("📈", "HIGHEST AQI",
         f"{valid.max():.1f}"),

        ("📊", "DATA RECORDS",
         f"{len(df):,}")
    ]

    cols = st.columns(4)

    for col, item in zip(cols, metrics_data):

        with col:

            st.markdown(f"""
            <div class="metric-card">
                <div>
                    {item[0]}
                    <span class="metric-label">{item[1]}</span>
                </div>

                <div class="metric-value">
                    {item[2]}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.write("")

    left, right = st.columns([1.5, 1])

    # AQI Distribution
    with left:

        st.markdown("### AQI Distribution")

        fig = px.histogram(
            df,
            x="AQI",
            nbins=35,
            color_discrete_sequence=["#13866b"]
        )

        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(
                l=10, r=10, t=20, b=10
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # AQI Categories
    with right:

        st.markdown("### Air Quality Categories")

        temp = df.copy()

        temp["Category"] = pd.cut(
            temp["AQI"],
            bins=[
                -1, 50, 100, 200,
                300, 400, 500
            ],
            labels=[
                "Good",
                "Satisfactory",
                "Moderate",
                "Poor",
                "Very Poor",
                "Severe"
            ]
        )

        counts = (
            temp["Category"]
            .value_counts()
            .reindex(
                [
                    "Good",
                    "Satisfactory",
                    "Moderate",
                    "Poor",
                    "Very Poor",
                    "Severe"
                ],
                fill_value=0
            )
            .reset_index()
        )

        counts.columns = ["Category", "Count"]

        fig = px.pie(
            counts,
            names="Category",
            values="Count",
            hole=0.65,
            color="Category",
            color_discrete_map={
                "Good": "#4CAF50",
                "Satisfactory": "#A8D672",
                "Moderate": "#FFC107",
                "Poor": "#FF7043",
                "Very Poor": "#AB47BC",
                "Severe": "#7E2430"
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # Recent records
    st.markdown("### Recent Environmental Records")

    recent = df.sort_values(
        "Date", na_position="first"
    ).tail(10)

    st.dataframe(
        recent,
        use_container_width=True
    )


# ==========================================
# PAGE 2: AQI PREDICTION
# ==========================================

elif page == "Predict AQI":

    st.markdown("### Air Quality Prediction")

    st.caption(
        "Enter pollutant concentrations to estimate AQI."
    )

    with st.form("aqi_form"):

        st.markdown("#### Environmental Parameters")

        col1, col2, col3 = st.columns(3)

        values = {}

        for i, feature in enumerate(features):

            with [col1, col2, col3][i % 3]:

                default_value = float(
                    medians[feature]
                )

                values[feature] = st.number_input(
                    feature,
                    min_value=0.0,
                    value=default_value,
                    step=0.1
                )

        submit = st.form_submit_button(
            "Generate AQI Prediction",
            use_container_width=True
        )

    if submit:

        input_df = pd.DataFrame(
            [values],
            columns=features
        )

        # Predict AQI
        result = float(
            model.predict(input_df)[0]
        )

        result = float(
            np.clip(result, 0, 500)
        )

        label, emoji = category(result)

        st.divider()

        a, b = st.columns(2)

        # Prediction result
        with a:

            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">
                    PREDICTED AIR QUALITY INDEX
                </div>

                <div class="metric-value">
                    {result:.2f}
                </div>

                <h3>{emoji} {label}</h3>
            </div>
            """, unsafe_allow_html=True)

        # AQI gauge
        with b:

            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=result,

                    gauge={
                        "axis": {
                            "range": [0, 500]
                        },

                        "bar": {
                            "color": "#13866b"
                        },

                        "steps": [
                            {
                                "range": [0, 50],
                                "color": "#b7e8c8"
                            },
                            {
                                "range": [50, 100],
                                "color": "#e9edb2"
                            },
                            {
                                "range": [100, 200],
                                "color": "#f6d49b"
                            },
                            {
                                "range": [200, 300],
                                "color": "#f3b1a8"
                            },
                            {
                                "range": [300, 400],
                                "color": "#c8b2e4"
                            },
                            {
                                "range": [400, 500],
                                "color": "#a9a9a9"
                            }
                        ]
                    }
                )
            )

            fig.update_layout(
                height=240,
                margin=dict(
                    l=15, r=15, t=20, b=10
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # Health recommendation
        st.markdown("### 🩺 Health Recommendation")

        st.info(advice[label])

        st.caption(
            "Model estimate based on historical data; "
            "not an official live AQI reading."
        )


# ==========================================
# PAGE 3: CITY EXPLORER
# ==========================================

elif page == "City Explorer":

    st.markdown("### Explore Indian Cities")

    cities = sorted(
        df["City"].dropna().unique()
    )

    selected = st.selectbox(
        "Select a city",
        cities
    )

    city_df = df[
        df["City"] == selected
    ].copy()

    city_df = city_df.dropna(
        subset=["Date", "AQI"]
    )

    if not city_df.empty:

        average_aqi = city_df["AQI"].mean()

        city_category, city_emoji = category(
            average_aqi
        )

        c1, c2 = st.columns(2)

        c1.metric(
            "Average AQI",
            f"{average_aqi:.2f}"
        )

        c2.metric(
            "Air Quality",
            f"{city_emoji} {city_category}"
        )

        st.markdown("### Historical AQI Trend")

        fig = px.line(
            city_df.sort_values("Date"),
            x="Date",
            y="AQI",
            title=f"Historical AQI - {selected}",
            color_discrete_sequence=["#13866b"]
        )

        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.warning(
            "No historical AQI records available for this city."
        )

    # City comparison
    st.markdown("### City Comparison")

    comparison = (
        df.groupby("City")["AQI"]
        .mean()
        .reset_index()
    )

    comparison = comparison.sort_values(
        "AQI", ascending=False
    )

    fig = px.bar(
        comparison,
        x="City",
        y="AQI",
        color="AQI",
        color_continuous_scale="Tealgrn",
        title="Average AQI by City"
    )

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ==========================================
# PAGE 4: POLLUTION INSIGHTS
# ==========================================

elif page == "Pollution Insights":

    st.markdown("### Pollution Analytics")

    available = [
        f for f in features
        if f in df.columns
    ]

    selected = st.selectbox(
        "Choose pollutant",
        available
    )

    # Pollutant distribution
    st.markdown("### Pollutant Distribution")

    fig = px.histogram(
        df,
        x=selected,
        nbins=40,
        title=f"{selected} Concentration Distribution",
        color_discrete_sequence=["#13866b"]
    )

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # Pollutant statistics
    st.markdown("### Pollutant Statistics")

    st.metric(
        f"Average {selected}",
        f"{df[selected].mean():.2f}"
        if df[selected].notna().any()
        else "N/A"
    )

    # Correlation analysis
    st.markdown("### Pollutant Correlation")

    corr = df[
        available + ["AQI"]
    ].corr(numeric_only=True)

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="Teal",
        title="Environmental Parameter Correlation"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ==========================================
# PAGE 5: MODEL LAB
# ==========================================

elif page == "Model Lab":

    st.markdown("### Machine Learning Laboratory")

    st.success(
        f"Best Selected Model: {best_name}"
    )

    metrics_path = "models/model_metrics.pkl"

    if os.path.exists(metrics_path):

        metrics = load_metrics()

        # Model evaluation table
        st.markdown("### Model Evaluation Results")

        st.dataframe(
            metrics,
            use_container_width=True
        )

        # R2 comparison
        st.markdown("### Algorithm Performance Comparison")

        fig = px.bar(
            metrics,
            x="Model",
            y="R2 Score",
            color="Model",
            title="R² Score Comparison"
        )

        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # RMSE comparison
        st.markdown("### Prediction Error Comparison")

        fig = px.bar(
            metrics,
            x="Model",
            y="RMSE",
            color="Model",
            title="RMSE Comparison"
        )

        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # Feature importance
    st.divider()

    st.markdown("### 🔍 Feature Importance")

    try:

        trained_algorithm = model.named_steps["model"]

        importance_values = (
            trained_algorithm.feature_importances_
        )

        importance = pd.DataFrame({
            "Feature": features,
            "Importance": importance_values
        })

        importance = importance.sort_values(
            "Importance",
            ascending=True
        )

        fig = px.bar(
            importance,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Tealgrn",
            title="Environmental Feature Importance"
        )

        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

        st.caption(
            "Feature importance represents the relative "
            "contribution of input features within the "
            "trained tree-based model. It does not prove "
            "causation."
        )

    except (AttributeError, KeyError):

        st.warning(
            "Feature importance is not available for this model."
        )

    # Project information
    st.divider()

    st.markdown("### Project Information")

    st.write("**Project Name:** AeroSense")

    st.write("**Best Algorithm:**", best_name)

    st.write("**Number of Features:**", len(features))

    st.write("**Learning Type:** Supervised Learning")

    st.write("**Problem Type:** Regression")

    st.write("**Dataset:** India Air Quality Dataset")


# ==========================================
# 10. FOOTER
# ==========================================

st.divider()

st.caption(
    "AeroSense | Air Quality Intelligence Platform | "
    "Python + Machine Learning + Streamlit"
)
