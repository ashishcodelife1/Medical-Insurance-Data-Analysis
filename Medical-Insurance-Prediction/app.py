"""MEDINSIGHT AI - Medical Insurance Cost Prediction System (Streamlit).

Run from the project root:   python -m streamlit run app.py
"""
import os
import traceback
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.model_utils import (
    CATEGORY_VALUES,
    DATA_PATH,
    DATASET_HELP,
    FEATURE_COLUMNS,
    METRICS_PATH,
    MODEL_PATH,
    DatasetError,
    ModelNotFoundError,
    load_dataset,
    load_metrics,
    load_model,
    predict_charge,
)

st.set_page_config(page_title="MEDINSIGHT AI", layout="wide", initial_sidebar_state="collapsed")

ROOT = Path(__file__).resolve().parent
CSS_PATH = ROOT / "assets" / "custom.css"
DEBUG = os.environ.get("MEDINSIGHT_DEBUG", "0") == "1"  # set to 1 to see stack traces

NAVY, TEAL, CORAL = "#0B2545", "#13A89E", "#E4572E"
PAGES = ["Home", "Prediction", "Model Performance", "Data Explorer", "About", "How It Works"]
NOTICE = "Educational estimate only. Not an actual insurance quote."

AGE_MIN, AGE_MAX = 18, 64
BMI_MIN, BMI_MAX = 15.0, 55.0
CHILD_MIN, CHILD_MAX = 0, 5
DEFAULTS = {
    "in_age": 30, "in_sex": "male", "in_bmi": 25.0,
    "in_children": 0, "in_smoker": "no", "in_region": "southwest",
}


# ------------------------------------------------------------- helpers ----
def load_css():
    if CSS_PATH.exists():
        st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def file_mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def usd(x: float) -> str:
    return f"${x:,.2f}"


def card(label, value, note=""):
    st.markdown(
        f'<div class="mi-card"><div class="mi-label">{label}</div>'
        f'<div class="mi-value">{value}</div><div class="mi-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def style_fig(fig, title):
    fig.update_layout(
        title=title, template="plotly_white", margin=dict(l=10, r=10, t=50, b=10),
        font=dict(family="Segoe UI, Arial, sans-serif", color=NAVY), height=380,
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def go_to(page: str):
    st.session_state["page"] = page


def reset_inputs():
    for key, value in DEFAULTS.items():
        st.session_state[key] = value
    st.session_state.pop("result", None)


# --------------------------------------------------------- data loading ----
@st.cache_data(show_spinner=False)
def _cached_dataset(_mtime: float):
    return load_dataset()


@st.cache_resource(show_spinner=False)
def _cached_model(_mtime: float):
    return load_model()


@st.cache_data(show_spinner=False)
def _cached_metrics(_mtime: float):
    return load_metrics()


def safe_load(loader, path):
    """Return (object, error_message). Never raises to the user."""
    try:
        return loader(file_mtime(path)), None
    except (DatasetError, ModelNotFoundError) as exc:
        return None, str(exc)
    except Exception as exc:
        if DEBUG:
            st.code(traceback.format_exc())
        return None, f"Unexpected problem while loading '{path.name}': {exc}"


# ----------------------------------------------------------- page parts ----
def render_header():
    st.markdown(
        f'<div class="mi-header"><div><div class="mi-brand">MEDINSIGHT <span>AI</span></div>'
        f'<div class="mi-sub">Medical Insurance Cost Prediction System</div></div>'
        f'<div class="mi-pill">{NOTICE}</div></div>',
        unsafe_allow_html=True,
    )


def dataset_missing_box(err):
    st.error(f"Dataset problem: {err}")
    st.info(
        f"{DATASET_HELP}\n\nExpected location: `{DATA_PATH}`. "
        "Source: Kaggle 'Medical Cost Personal Datasets' (insurance.csv). See README.md for steps."
    )


def model_missing_box(err):
    st.warning(f"{err}")
    st.code("python train_model.py", language="bash")


def page_home(df, df_err, metrics, m_err):
    st.markdown(
        '<div class="mi-hero"><h1>Estimate medical insurance charges with Machine Learning</h1>'
        "<p>MEDINSIGHT AI learns from historical insurance records and predicts the annual "
        "charges (in US dollars) for a person's age, sex, BMI, children, smoking status and region.</p></div>",
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        if df is not None:
            card("Dataset size", f"{len(df):,} records", f"{df.shape[1]} columns loaded from insurance.csv")
        else:
            card("Dataset size", "Not loaded", "insurance.csv missing or invalid")
    with c2:
        card("Model type", "Linear Regression", "Scikit-learn pipeline with OneHotEncoder")
    with c3:
        if metrics is not None:
            card("Evaluation status", "Evaluated", f"Test R\u00b2 = {metrics['r2']:.3f} (from metrics.json)")
        else:
            card("Evaluation status", "Not trained yet", "Run train_model.py to evaluate")

    st.write("")
    st.button("Start a prediction", type="primary", on_click=go_to, args=("Prediction",))

    st.markdown("### What does it predict?")
    st.write(
        "The target is **charges**: the individual medical costs billed by health insurance for one year, "
        "as recorded in the historical dataset. The output is a statistical estimate, not a quotation."
    )
    st.markdown("### Algorithm")
    st.write(
        "**Linear Regression** is a supervised learning algorithm that fits a straight-line (linear) relationship "
        "between the input features and the numeric target."
    )
    st.markdown("### Dataset summary")
    if df is None:
        dataset_missing_box(df_err)
    else:
        st.write(
            f"Features: **{', '.join(FEATURE_COLUMNS)}**. Target: **charges**. "
            f"Mean charge in the loaded data: **{usd(df['charges'].mean())}**."
        )
    if m_err and df is not None:
        model_missing_box(m_err)


def validate_inputs(v):
    problems = []
    if v["age"] is None or not (AGE_MIN <= v["age"] <= AGE_MAX):
        problems.append(f"Age must be between {AGE_MIN} and {AGE_MAX}.")
    if v["bmi"] is None or not (BMI_MIN <= v["bmi"] <= BMI_MAX):
        problems.append(f"BMI must be between {BMI_MIN} and {BMI_MAX}.")
    if v["children"] is None or not (CHILD_MIN <= v["children"] <= CHILD_MAX):
        problems.append(f"Children must be between {CHILD_MIN} and {CHILD_MAX}.")
    for col in ("sex", "smoker", "region"):
        if v[col] not in CATEGORY_VALUES[col]:
            problems.append(f"Please choose a valid value for {col}.")
    return problems


def page_prediction():
    model, model_err = safe_load(_cached_model, MODEL_PATH)
    if model is None:
        model_missing_box(model_err)
        return

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("### Enter details")
        st.caption("Values are used only for this prediction and are not stored.")
        a, b = st.columns(2)
        with a:
            st.number_input("Age (years)", min_value=AGE_MIN, max_value=AGE_MAX, step=1, key="in_age",
                            help=f"Dataset ages range roughly {AGE_MIN}-{AGE_MAX}.")
            st.number_input("BMI (body mass index)", min_value=BMI_MIN, max_value=BMI_MAX, step=0.1,
                            format="%.1f", key="in_bmi", help="Weight (kg) divided by height (m) squared.")
            st.selectbox("Smoking status", CATEGORY_VALUES["smoker"], key="in_smoker",
                         format_func=str.capitalize, help="Does the person smoke?")
        with b:
            st.selectbox("Sex", CATEGORY_VALUES["sex"], key="in_sex", format_func=str.capitalize,
                         help="Sex as recorded in the dataset.")
            st.number_input("Number of children", min_value=CHILD_MIN, max_value=CHILD_MAX, step=1,
                            key="in_children", help="Children covered by the insurance plan.")
            st.selectbox("Region (US)", CATEGORY_VALUES["region"], key="in_region",
                         format_func=str.capitalize, help="Residential area in the US.")

        b1, b2 = st.columns(2)
        predict_clicked = b1.button("Predict Insurance Cost", type="primary", use_container_width=True)
        b2.button("Reset", on_click=reset_inputs, use_container_width=True)

        if predict_clicked:
            values = {
                "age": st.session_state.get("in_age"), "sex": st.session_state.get("in_sex"),
                "bmi": st.session_state.get("in_bmi"), "children": st.session_state.get("in_children"),
                "smoker": st.session_state.get("in_smoker"), "region": st.session_state.get("in_region"),
            }
            problems = validate_inputs(values)
            if problems:
                for p in problems:
                    st.error(p)
            else:
                try:
                    with st.spinner("Calculating estimate..."):
                        pred = predict_charge(model, values)
                    st.session_state["result"] = {"inputs": values, "prediction": pred}
                except Exception as exc:
                    if DEBUG:
                        st.code(traceback.format_exc())
                    st.error(f"Prediction failed: {exc}")

    with right:
        st.markdown("### Result")
        result = st.session_state.get("result")
        if result is None:
            st.markdown(
                '<div class="mi-empty">No prediction yet.<br>Fill in the form and press '
                "<b>Predict Insurance Cost</b>.</div>", unsafe_allow_html=True)
        else:
            pred = result["prediction"]
            shown = max(pred, 0.0)
            st.markdown(
                f'<div class="mi-result"><div class="mi-label">Estimated annual charges</div>'
                f'<div class="mi-big">{usd(shown)}</div>'
                f'<div class="mi-note">US dollars (USD) per year</div></div>', unsafe_allow_html=True)
            if pred < 0:
                st.warning("The raw model output was negative, which is unrealistic. "
                           "These inputs are outside the range the model handles well; shown as $0.00.")
            summary = pd.DataFrame(
                {"Input": ["Age", "Sex", "BMI", "Children", "Smoker", "Region"],
                 "Value": [result["inputs"]["age"], str(result["inputs"]["sex"]).capitalize(),
                           f'{result["inputs"]["bmi"]:.1f}', result["inputs"]["children"],
                           str(result["inputs"]["smoker"]).capitalize(),
                           str(result["inputs"]["region"]).capitalize()]})
            st.table(summary.set_index("Input"))
            st.write(
                "**What this means:** the Linear Regression model combined the six inputs using patterns "
                "learned from historical records to produce one estimated yearly charge.")
            st.markdown(
                '<div class="mi-disclaimer"><b>Disclaimer:</b> This is a statistical estimate based on '
                "historical data. It is not an insurance quotation, does not diagnose any medical condition, "
                "and learned associations do not prove causation. Do not use it alone for real insurance decisions."
                "</div>", unsafe_allow_html=True)


def page_performance(metrics, m_err):
    st.markdown("### Model performance (held-out test set)")
    if metrics is None:
        model_missing_box(m_err)
        return
    c1, c2, c3 = st.columns(3)
    with c1:
        card("Mean Absolute Error", usd(metrics["mae"]), "Average size of the error")
    with c2:
        card("Root Mean Squared Error", usd(metrics["rmse"]), "Error that punishes big misses")
    with c3:
        card("R\u00b2 score", f"{metrics['r2']:.4f}", "Share of variation explained")
    st.caption(
        f"Trained on {metrics['train_rows']:,} rows, tested on {metrics['test_rows']:,} unseen rows "
        f"(test size {int(metrics['test_size'] * 100)}%, random seed {metrics['random_state']}). "
        f"Trained at {metrics['trained_at']}.")

    tp = pd.DataFrame(metrics["test_predictions"])
    fig = px.scatter(tp, x="actual", y="predicted", opacity=0.65, color_discrete_sequence=[TEAL],
                     labels={"actual": "Actual charges (USD)", "predicted": "Predicted charges (USD)"})
    lo, hi = float(min(tp.min())), float(max(tp.max()))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect prediction",
                             line=dict(color=NAVY, dash="dash")))
    st.plotly_chart(style_fig(fig, "Actual vs predicted charges (test set)"), use_container_width=True)

    st.markdown("### Plain-language interpretation")
    st.markdown(
        f"- **MAE ({usd(metrics['mae'])})**: on average, predictions differ from the real charge by about this much.\n"
        f"- **RMSE ({usd(metrics['rmse'])})**: similar to MAE, but large mistakes count more. "
        "If RMSE is much bigger than MAE, some predictions are far off.\n"
        f"- **R\u00b2 ({metrics['r2']:.3f})**: the model explains about {metrics['r2'] * 100:.1f}% of the variation "
        "in charges on the test data. 1.0 is perfect; 0 is no better than always guessing the average. "
        "R\u00b2 is a regression score, **not** classification accuracy.\n"
        "- Points close to the dashed line in the chart are accurate predictions.")


def page_explorer(df, df_err):
    st.markdown("### Data explorer")
    if df is None:
        dataset_missing_box(df_err)
        return
    a, b = st.columns(2)
    with a:
        card("Rows", f"{df.shape[0]:,}", "Records in the dataset")
    with b:
        card("Columns", f"{df.shape[1]}", ", ".join(df.columns))

    st.markdown("#### Dataset preview")
    st.dataframe(df.head(20), use_container_width=True)

    t1, t2 = st.columns(2)
    with t1:
        st.markdown("#### Column data types")
        st.dataframe(pd.DataFrame({"column": df.columns, "dtype": [str(t) for t in df.dtypes]}),
                     use_container_width=True, hide_index=True)
    with t2:
        st.markdown("#### Missing values")
        st.dataframe(pd.DataFrame({"column": df.columns, "missing": df.isna().sum().values}),
                     use_container_width=True, hide_index=True)

    st.markdown("#### Descriptive statistics")
    st.dataframe(df.describe().T, use_container_width=True)

    colors = {"yes": CORAL, "no": TEAL}
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x="charges", nbins=40, color_discrete_sequence=[NAVY],
                           labels={"charges": "Charges (USD)"})
        st.plotly_chart(style_fig(fig, "Distribution of insurance charges"), use_container_width=True)
    with c2:
        fig = px.scatter(df, x="bmi", y="charges", color="smoker", color_discrete_map=colors, opacity=0.65,
                         labels={"bmi": "BMI", "charges": "Charges (USD)", "smoker": "Smoker"})
        st.plotly_chart(style_fig(fig, "BMI vs insurance charges"), use_container_width=True)
    c3, c4 = st.columns(2)
    with c3:
        fig = px.scatter(df, x="age", y="charges", color="smoker", color_discrete_map=colors, opacity=0.65,
                         labels={"age": "Age (years)", "charges": "Charges (USD)", "smoker": "Smoker"})
        st.plotly_chart(style_fig(fig, "Age vs insurance charges"), use_container_width=True)
    with c4:
        fig = px.box(df, x="smoker", y="charges", color="smoker", color_discrete_map=colors,
                     labels={"smoker": "Smoker", "charges": "Charges (USD)"})
        st.plotly_chart(style_fig(fig, "Smoking status vs insurance charges"), use_container_width=True)
    st.caption("Charts show associations in historical data only; they do not prove cause and effect.")


def page_about():
    st.markdown("### About the project")
    st.markdown("**Problem statement**")
    st.write("Insurance charges differ between people. This project builds a regression model that estimates "
             "annual medical insurance charges from six personal attributes using historical data.")
    st.markdown("**Objectives**")
    st.markdown("- Apply supervised learning (Linear Regression) to a healthcare case study.\n"
                "- Practise preprocessing, categorical encoding, train/test splitting and evaluation.\n"
                "- Save the trained pipeline and serve predictions through a web interface.")
    st.markdown("**Machine Learning algorithm**")
    st.write("Linear Regression inside a scikit-learn Pipeline, with OneHotEncoder for sex, smoker and region.")
    st.markdown("**Dataset features**")
    st.markdown("- `age`, `bmi`, `children` (numeric)\n- `sex`, `smoker`, `region` (categorical)\n"
                "- `charges` (target, USD)")
    st.markdown("**Technologies used**")
    st.write("Python, Streamlit, Scikit-learn, Pandas, NumPy, Plotly, Joblib, VS Code.")
    st.markdown("**Limitations**")
    st.markdown("- Linear Regression assumes straight-line relationships and may miss interactions.\n"
                "- The dataset is small and covers a limited population (US data, USD).\n"
                "- Predictions may differ on new populations or later years.")
    st.markdown("**Ethical considerations**")
    st.markdown("- Historical data can contain bias; the model may reproduce it.\n"
                "- Associations are not causation.\n"
                "- Not an insurance quote, not a diagnosis, not for real insurance decisions on its own.\n"
                "- No personal data is collected or stored.")
    st.markdown(f'<div class="mi-disclaimer">{NOTICE}</div>', unsafe_allow_html=True)


def page_how():
    st.markdown("### How it works")
    steps = ["Historical Insurance Dataset", "Data Loading", "Data Validation and Preprocessing",
             "Training and Testing Split", "Linear Regression Training", "Model Evaluation",
             "Save Trained Pipeline", "User Input", "Prediction", "Display Estimated Insurance Charges"]
    html = "".join(f'<div class="mi-step"><div class="mi-num">{i}</div>'
                   f'<div class="mi-step-text">{s}</div></div>' for i, s in enumerate(steps, 1))
    st.markdown(f'<div class="mi-flow">{html}</div>', unsafe_allow_html=True)
    st.write("Steps 1-7 happen once when you run `train_model.py`. Steps 8-10 happen each time you press "
             "**Predict Insurance Cost**; the saved pipeline is loaded, never retrained.")


# ----------------------------------------------------------------- main ----
def main():
    load_css()
    for key, value in DEFAULTS.items():
        st.session_state.setdefault(key, value)
    st.session_state.setdefault("page", "Home")

    render_header()
    st.radio("Navigation", PAGES, key="page", horizontal=True, label_visibility="collapsed")

    df, df_err = safe_load(_cached_dataset, DATA_PATH)
    metrics, m_err = safe_load(_cached_metrics, METRICS_PATH)

    page = st.session_state["page"]
    if page == "Home":
        page_home(df, df_err, metrics, m_err)
    elif page == "Prediction":
        page_prediction()
    elif page == "Model Performance":
        page_performance(metrics, m_err)
    elif page == "Data Explorer":
        page_explorer(df, df_err)
    elif page == "About":
        page_about()
    else:
        page_how()
    st.caption(NOTICE)


main()
