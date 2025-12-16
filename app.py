import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# =========================
# TEXT (BILINGUAL)
# =========================
TEXT = {
    "en": {
        "title": "Survey Web App: Descriptive & Association (Pearson)",
        "sidebar": "Settings",
        "lang": "Language",
        "upload": "Upload dataset (CSV or XLSX)",
        "no_file": "No file uploaded yet.",
        "tabs": ["Overview", "Descriptive", "Visualization", "Association"],
        "preview": "Data preview (first 5 rows)",

        "desc_title": "Descriptive Analysis",
        "choose_desc_cols": "Select numeric columns for descriptive statistics",
        "desc_table": "Descriptive statistics (numeric)",

        "viz_title": "Visualization",
        "viz_note": "Choose a numeric column to visualize distribution.",
        "choose_viz_col": "Select numeric column",
        "viz_type": "Chart type",
        "bar": "Bar chart (value counts)",
        "hist": "Histogram",

        "assoc_title": "Association Analysis (Pearson)",
        "assoc_note": "Pearson correlation between two numeric variables.",
        "x_var": "Select X variable",
        "y_var": "Select Y variable",
        "run": "Run Pearson Correlation",
        "pearson_r": "Pearson r",
        "p_value": "p-value",
        "interpret": "Interpretation",
        "direction": "Direction",
        "strength": "Strength",
        "sig": "Significance",
        "significant": "Significant (p < 0.05)",
        "not_significant": "Not significant (p ≥ 0.05)",
        "pos": "Positive",
        "neg": "Negative",
        "weak": "Weak",
        "moderate": "Moderate",
        "strong": "Strong",
        "scatter_reg": "Scatter + regression line",
        "error_numeric": "Selected columns must be numeric.",
        "error_pairs": "Not enough valid pairs (need ≥ 3 rows after removing missing).",
    },
    "id": {
        "title": "Web Survei: Deskriptif & Asosiasi (Pearson)",
        "sidebar": "Pengaturan",
        "lang": "Bahasa",
        "upload": "Unggah dataset (CSV atau XLSX)",
        "no_file": "Belum ada file yang diunggah.",
        "tabs": ["Ringkasan", "Deskriptif", "Visualisasi", "Asosiasi"],
        "preview": "Pratinjau data (5 baris pertama)",

        "desc_title": "Analisis Deskriptif",
        "choose_desc_cols": "Pilih kolom numerik untuk statistik deskriptif",
        "desc_table": "Statistik deskriptif (numerik)",

        "viz_title": "Visualisasi",
        "viz_note": "Pilih 1 kolom numerik untuk melihat distribusi.",
        "choose_viz_col": "Pilih kolom numerik",
        "viz_type": "Jenis grafik",
        "bar": "Grafik batang (frekuensi)",
        "hist": "Histogram",

        "assoc_title": "Analisis Asosiasi (Pearson)",
        "assoc_note": "Korelasi Pearson antara dua variabel numerik.",
        "x_var": "Pilih variabel X",
        "y_var": "Pilih variabel Y",
        "run": "Jalankan Korelasi Pearson",
        "pearson_r": "Pearson r",
        "p_value": "Nilai p",
        "interpret": "Interpretasi",
        "direction": "Arah",
        "strength": "Kekuatan",
        "sig": "Signifikansi",
        "significant": "Signifikan (p < 0.05)",
        "not_significant": "Tidak signifikan (p ≥ 0.05)",
        "pos": "Positif",
        "neg": "Negatif",
        "weak": "Lemah",
        "moderate": "Sedang",
        "strong": "Kuat",
        "scatter_reg": "Scatter + garis regresi",
        "error_numeric": "Kolom yang dipilih harus numerik.",
        "error_pairs": "Pasangan data valid tidak cukup (minimal ≥ 3 baris setelah hapus missing).",
    },
}

# =========================
# DARK MODE CSS (ALWAYS)
# =========================
def apply_dark_mode():
    st.markdown(
        """
        <style>
        .stApp { background: #0b1220; }
        [data-testid="stSidebar"] { background: #0a0f1a; }
        h1,h2,h3,h4,h5,h6,p,li,div,span,label { color: #e8eefc !important; }
        .block-container { padding-top: 2rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# =========================
# HELPERS
# =========================
def to_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def strength_of_r(r: float, lang: str) -> str:
    ar = abs(r)
    if ar < 0.30:
        return TEXT[lang]["weak"]
    elif ar < 0.70:
        return TEXT[lang]["moderate"]
    return TEXT[lang]["strong"]

def direction_of_r(r: float, lang: str) -> str:
    if r > 0:
        return TEXT[lang]["pos"]
    return TEXT[lang]["neg"]

def fig_bar_counts(series: pd.Series, title: str):
    s = series.dropna()
    counts = s.value_counts().sort_index()
    fig, ax = plt.subplots()
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    return fig

def fig_hist(series: pd.Series, title: str):
    s = series.dropna()
    fig, ax = plt.subplots()
    ax.hist(s, bins=5)
    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    return fig

def fig_scatter_reg(x: pd.Series, y: pd.Series, title: str):
    df_xy = pd.DataFrame({"x": x, "y": y}).dropna()
    x = df_xy["x"]
    y = df_xy["y"]

    fig, ax = plt.subplots()
    ax.scatter(x, y)

    slope, intercept, r_val, p_val, se = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = intercept + slope * x_line
    ax.plot(x_line, y_line)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    return fig

# =========================
# MAIN
# =========================
def main():
    apply_dark_mode()

    # Sidebar language
    st.sidebar.header("⚙️ Settings")
    lang_pick = st.sidebar.selectbox("Language / Bahasa", ["English", "Bahasa Indonesia"])
    lang = "id" if lang_pick == "Bahasa Indonesia" else "en"
    T = TEXT[lang]

    st.title(T["title"])

    uploaded = st.sidebar.file_uploader(T["upload"], type=["csv", "xlsx"])
    if uploaded is None:
        st.info(T["no_file"])
        return

    # Load data
    if uploaded.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)

    tab1, tab2, tab3, tab4 = st.tabs(T["tabs"])

    # =========================
    # TAB 1: OVERVIEW (minimal)
    # =========================
    with tab1:
        st.subheader(T["tabs"][0])
        st.write(f"Rows: **{df.shape[0]}** | Columns: **{df.shape[1]}**")
        st.markdown("#### " + T["preview"])
        st.dataframe(df.head())

    # =========================
    # TAB 2: DESCRIPTIVE
    # =========================
    with tab2:
        st.subheader(T["desc_title"])

        # detect numeric-like columns
        numeric_like = []
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                numeric_like.append(c)
            else:
                sample = pd.to_numeric(df[c], errors="coerce")
                if sample.notna().mean() >= 0.5:
                    numeric_like.append(c)

        selected = st.multiselect(T["choose_desc_cols"], options=numeric_like)

        if selected:
            temp = df[selected].copy()
            for c in selected:
                temp[c] = to_numeric_series(temp[c])
            st.markdown("#### " + T["desc_table"])
            st.dataframe(temp.describe().T.round(4))

    # =========================
    # TAB 3: VISUALIZATION (Bar + Histogram only)
    # =========================
    with tab3:
        st.subheader(T["viz_title"])
        st.caption(T["viz_note"])

        numeric_like = []
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                numeric_like.append(c)
            else:
                sample = pd.to_numeric(df[c], errors="coerce")
                if sample.notna().mean() >= 0.5:
                    numeric_like.append(c)

        if not numeric_like:
            st.warning("No numeric columns found.")
            return

        col = st.selectbox(T["choose_viz_col"], options=numeric_like)
        chart_type = st.radio(T["viz_type"], [T["bar"], T["hist"]], horizontal=True)

        s = to_numeric_series(df[col])

        if chart_type == T["bar"]:
            st.pyplot(fig_bar_counts(s, f"{col} - value counts"))
        else:
            st.pyplot(fig_hist(s, f"{col} - histogram"))

    # =========================
    # TAB 4: ASSOCIATION (PEARSON ONLY)
    # =========================
    with tab4:
        st.subheader(T["assoc_title"])
        st.caption(T["assoc_note"])

        x_var = st.selectbox(T["x_var"], options=df.columns.tolist())
        y_var = st.selectbox(T["y_var"], options=df.columns.tolist())

        if st.button(T["run"]):
            x = to_numeric_series(df[x_var])
            y = to_numeric_series(df[y_var])

            pair = pd.DataFrame({"x": x, "y": y}).dropna()

            if len(pair) < 3:
                st.error(T["error_pairs"])
                return

            r, p = stats.pearsonr(pair["x"], pair["y"])

            st.write(f"**{T['pearson_r']}:** {r:.4f}")
            st.write(f"**{T['p_value']}:** {p:.4f}")

            st.markdown("### " + T["interpret"])
            st.write(f"**{T['direction']}:** {direction_of_r(r, lang)}")
            st.write(f"**{T['strength']}:** {strength_of_r(r, lang)}")
            st.write(f"**{T['sig']}:** {T['significant'] if p < 0.05 else T['not_significant']}")

            st.markdown("### " + T["scatter_reg"])
            st.pyplot(fig_scatter_reg(pair["x"], pair["y"], f"{x_var} vs {y_var} | r={r:.4f}, p={p:.4f}"))

if __name__ == "__main__":
    main()