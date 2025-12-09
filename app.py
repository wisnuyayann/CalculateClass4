# app.py
# Aplikasi Streamlit untuk analisis data survei dengan fitur bilingual (English/Bahasa Indonesia).
# Menggunakan library: streamlit, pandas, numpy, scipy.

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats

# ===================== TEXT BILINGUAL ===================== #
texts = {
    'en': {
        'title': 'Survey Analysis: Social Media Use and Sleep Quality',
        'lang_select': 'Select Language / Pilih Bahasa',
        'upload': 'Upload Dataset (CSV or XLSX)',
        'preview': 'Data Preview (First 5 Rows)',
        'no_file': 'No file uploaded yet.',
        'desc_header': 'Descriptive Statistics',
        'assoc_header': 'Association Analysis',
        'assoc_note': 'Association analysis uses Pearson correlation between composite scores X_total and Y_total.',
        'demo_select': 'Select Demographic Columns',
        'demo_freq': 'Frequency Table for',
        'likert_select': 'Select Likert Scale Columns',
        'likert_stats': 'Descriptive Statistics for Likert Items',
        'assoc_x': 'Select Columns for X (e.g., Social Media Use Intensity)',
        'assoc_y': 'Select Columns for Y (e.g., Sleep Quality)',
        'assoc_results': 'Association Analysis Results',
        'correlation': 'Pearson Correlation (r):',
        'p_value': 'P-value:',
        'strength': 'Correlation Strength:',
        'direction': 'Relationship Direction:',
        'weak': 'Weak',
        'moderate': 'Moderate',
        'strong': 'Strong',
        'positive': 'Positive',
        'negative': 'Negative',
        'no_corr': 'Very weak or no correlation',
        'scatter_title': 'Scatter Plot: X_total vs Y_total',
        'error_numeric': 'Selected columns must be numeric for analysis.',
        'error_likert': 'Unable to extract numeric values from Likert items.',
        'error_not_enough': 'Not enough valid data to compute Pearson correlation (need at least 2 valid pairs).',
    },
    'id': {
        'title': 'Analisis Survei: Penggunaan Media Sosial dan Kualitas Tidur',
        'lang_select': 'Pilih Bahasa / Select Language',
        'upload': 'Unggah Dataset (CSV atau XLSX)',
        'preview': 'Pratinjau Data (5 Baris Pertama)',
        'no_file': 'Belum ada file yang diunggah.',
        'desc_header': 'Statistik Deskriptif',
        'assoc_header': 'Analisis Asosiasi',
        'assoc_note': 'Analisis asosiasi menggunakan korelasi Pearson antara skor komposit X_total dan Y_total.',
        'demo_select': 'Pilih Kolom Demografi',
        'demo_freq': 'Tabel Frekuensi untuk',
        'likert_select': 'Pilih Kolom Skala Likert',
        'likert_stats': 'Statistik Deskriptif untuk Item Likert',
        'assoc_x': 'Pilih Kolom untuk X (misalnya Intensitas Penggunaan Media Sosial)',
        'assoc_y': 'Pilih Kolom untuk Y (misalnya Kualitas Tidur)',
        'assoc_results': 'Hasil Analisis Asosiasi',
        'correlation': 'Korelasi Pearson (r):',
        'p_value': 'Nilai p:',
        'strength': 'Kekuatan Korelasi:',
        'direction': 'Arah Hubungan:',
        'weak': 'Lemah',
        'moderate': 'Sedang',
        'strong': 'Kuat',
        'positive': 'Positif',
        'negative': 'Negatif',
        'no_corr': 'Sangat lemah atau tidak ada korelasi',
        'scatter_title': 'Grafik Scatter: X_total vs Y_total',
        'error_numeric': 'Kolom yang dipilih harus numerik untuk analisis.',
        'error_likert': 'Tidak dapat mengekstrak nilai numerik dari item Likert.',
        'error_not_enough': 'Data valid tidak cukup untuk menghitung korelasi Pearson (minimal 2 pasang pengamatan).',
    }
}

# ===================== FUNGSI UTILITAS ===================== #

# Fungsi untuk mengekstrak angka dari teks Likert (misalnya "4 – Setuju" menjadi 4)
def extract_likert_value(value):
    """
    Mengembalikan nilai numerik (float) dari respon Likert.
    Menangani kasus:
      - sudah numerik (1, 2, 3, 4, 5 atau 4.0)
      - teks dengan angka di depan, misalnya "4 – Setuju" atau "3 - Netral"
    """
    if pd.isna(value):
        return np.nan

    # Jika sudah numerik (int/float), langsung kembalikan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip()

    # Coba langsung konversi ke float (misalnya "4", "4.0")
    try:
        return float(text)
    except ValueError:
        pass

    # Jika gagal, coba split berdasarkan " – " atau " - "
    for sep in [" – ", " - ", "-", "–"]:
        if sep in text:
            first_part = text.split(sep)[0].strip()
            try:
                return float(first_part)
            except ValueError:
                continue

    # Coba ambil digit pertama dalam teks
    for char in text:
        if char.isdigit():
            return float(char)

    return np.nan  # Jika semua gagal

# Fungsi untuk menghitung mode (karena pandas describe() tidak include mode)
def calculate_mode(series):
    mode_val = series.mode()
    return mode_val.iloc[0] if not mode_val.empty else np.nan

# Fungsi untuk interpretasi kekuatan korelasi
def interpret_strength(r):
    abs_r = abs(r)
    if abs_r < 0.3:
        return 'weak'
    elif abs_r < 0.7:
        return 'moderate'
    else:
        return 'strong'

# Fungsi untuk interpretasi arah korelasi
def interpret_direction(r):
    if r > 0:
        return 'positive'
    elif r < 0:
        return 'negative'
    else:
        return 'no_corr'

# ===================== APLIKASI UTAMA ===================== #

def main():
    # Sidebar: Pilihan bahasa
    lang = st.sidebar.selectbox(
        texts['en']['lang_select'],  # label bilingual di sini tidak terlalu penting
        ['English', 'Bahasa Indonesia']
    )
    lang_code = 'id' if lang == 'Bahasa Indonesia' else 'en'
    T = texts[lang_code]

    # Judul aplikasi
    st.title(T['title'])

    # Sidebar: Upload file
    uploaded_file = st.sidebar.file_uploader(T['upload'], type=['csv', 'xlsx'])

    if uploaded_file is not None:
        # Baca file berdasarkan ekstensi
        if uploaded_file.name.lower().endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # Preview data: 5 baris pertama
        st.header(T['preview'])
        st.dataframe(df.head())

        # ---------------- DESCRIPTIVE STATISTICS ---------------- #
        st.header(T['desc_header'])

        # Kolom demografi: Pilih kolom, tampilkan tabel frekuensi
        demo_cols = st.multiselect(T['demo_select'], df.columns.tolist())
        for col in demo_cols:
            if col in df.columns:
                freq = df[col].value_counts(dropna=False).reset_index()
                freq.columns = [col, 'Count']
                freq['Percentage'] = (freq['Count'] / len(df) * 100).round(2)
                st.subheader(f"{T['demo_freq']} {col}")
                st.dataframe(freq)

        # Kolom Likert: Pilih kolom, ekstrak angka, tampilkan stats
        likert_cols = st.multiselect(T['likert_select'], df.columns.tolist())
        likert_numeric = None
        if likert_cols:
            likert_numeric = df[likert_cols].applymap(extract_likert_value)
            likert_numeric = likert_numeric.astype(float)

            # Drop baris yang semuanya NaN
            likert_numeric = likert_numeric.dropna(how='all')

            if not likert_numeric.empty:
                desc = likert_numeric.describe().T[
                    ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
                ].round(2)
                desc['mode'] = likert_numeric.apply(calculate_mode)
                st.subheader(T['likert_stats'])
                st.dataframe(desc)
            else:
                st.error(T['error_likert'])

        # ---------------- ASSOCIATION ANALYSIS ---------------- #
        st.header(T['assoc_header'])
        st.caption(T['assoc_note'])

        # Pilih kolom untuk X dan Y
        x_cols = st.multiselect(T['assoc_x'], df.columns.tolist())
        y_cols = st.multiselect(T['assoc_y'], df.columns.tolist())

        if x_cols and y_cols:
            try:
                # Siapkan data X
                x_raw = df[x_cols]
                x_conv = x_raw.applymap(extract_likert_value).astype(float)

                # Siapkan data Y
                y_raw = df[y_cols]
                y_conv = y_raw.applymap(extract_likert_value).astype(float)

                # Hitung skor komposit: rata-rata per responden
                x_total = x_conv.mean(axis=1)
                y_total = y_conv.mean(axis=1)

                # Buang pasangan yang mengandung NaN
                mask = x_total.notna() & y_total.notna()
                x_valid = x_total[mask]
                y_valid = y_total[mask]

                if len(x_valid) < 2:
                    st.error(T['error_not_enough'])
                else:
                    # Pearson correlation
                    r, p = stats.pearsonr(x_valid, y_valid)

                    # Interpretasi
                    strength_key = interpret_strength(r)
                    direction_key = interpret_direction(r)

                    # Tampilkan hasil
                    st.subheader(T['assoc_results'])
                    st.write(f"{T['correlation']} {r:.3f}")
                    st.write(f"{T['p_value']} {p:.4f}")
                    st.write(f"{T['strength']} {T[strength_key]}")
                    st.write(f"{T['direction']} {T[direction_key]}")

                    # Scatter plot
                    scatter_df = pd.DataFrame({
                        'X_total': x_valid,
                        'Y_total': y_valid
                    })
                    st.subheader(T['scatter_title'])
                    st.scatter_chart(scatter_df, x='X_total', y='Y_total')

            except Exception as e:
                st.error(f"{T['error_numeric']} Error: {str(e)}")

    else:
        st.info(T['no_file'])


if __name__ == '__main__':
    main()