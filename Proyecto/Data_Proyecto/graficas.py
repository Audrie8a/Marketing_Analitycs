# resumen_totales_creditos.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

CSV_PATH = "dummy_creditos.csv"   # <- ajusta si tu archivo se llama distinto

# ------------------ utilidades ------------------
def leer_df(path):
    df = pd.read_csv(path)
    # normalizar
    df["saldo"] = pd.to_numeric(df["saldo"], errors="coerce").fillna(0.0)
    df["clasificacion"] = df["clasificacion"].astype(str).str.upper().str.strip()
    # orden A-E
    orden = pd.CategoricalDtype(categories=["A","B","C","D","E"], ordered=True)
    df["clasificacion"] = df["clasificacion"].astype(orden)
    return df

def plot_barh_series(serie, titulo, xlabel, ylabel):
    serie = serie.sort_values()
    plt.figure()
    plt.barh(serie.index.astype(str), serie.values)
    plt.title(titulo)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()

def plot_stacked_bar(pvt, titulo, xlabel, ylabel):
    # pvt: index = categorías eje X, columns = series apiladas
    pvt = pvt.fillna(0)
    x = np.arange(len(pvt.index))
    bottom = np.zeros(len(pvt.index))
    plt.figure()
    for col in pvt.columns:
        vals = pvt[col].values
        plt.bar(pvt.index.astype(str), vals, bottom=bottom, label=str(col))
        bottom += vals
    plt.title(titulo)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(fontsize=8, title=pvt.columns.name if pvt.columns.name else None)
    plt.tight_layout()

def head_index(df, col, n=10, by="saldo"):
    return df.groupby(col)[by].sum().sort_values(ascending=False).head(n).index

# ------------------ reportes ------------------
def totales_simples(df):
    # Clasificación
    plot_barh_series(df.groupby("clasificacion")["codigo_cuentahabiente"].size(),
                     "Conteo por CLASIFICACIÓN", "Conteo", "Clasificación")
    plot_barh_series(df.groupby("clasificacion")["saldo"].sum(),
                     "Saldo total por CLASIFICACIÓN", "Saldo total", "Clasificación")

    # Segmento
    plot_barh_series(df.groupby("segmento")["codigo_cuentahabiente"].size(),
                     "Conteo por SEGMENTO", "Conteo", "Segmento")
    plot_barh_series(df.groupby("segmento")["saldo"].sum(),
                     "Saldo total por SEGMENTO", "Saldo total", "Segmento")

    # Situación
    plot_barh_series(df.groupby("situacion")["codigo_cuentahabiente"].size(),
                     "Conteo por SITUACIÓN", "Conteo", "Situación")
    plot_barh_series(df.groupby("situacion")["saldo"].sum(),
                     "Saldo total por SITUACIÓN", "Saldo total", "Situación")

    # Tipo de deudor
    plot_barh_series(df.groupby("tipo_deudor")["codigo_cuentahabiente"].size(),
                     "Conteo por TIPO DE DEUDOR", "Conteo", "Tipo de deudor")
    plot_barh_series(df.groupby("tipo_deudor")["saldo"].sum(),
                     "Saldo total por TIPO DE DEUDOR", "Saldo total", "Tipo de deudor")

def combinaciones(df):
    # Clasificación × Segmento (conteo y saldo)
    pvt_cnt = pd.pivot_table(df, index="segmento", columns="clasificacion",
                             values="codigo_cuentahabiente", aggfunc="size", fill_value=0)
    plot_stacked_bar(pvt_cnt, "Conteo por SEGMENTO × CLASIFICACIÓN", "Segmento", "Conteo")

    pvt_sum = pd.pivot_table(df, index="segmento", columns="clasificacion",
                             values="saldo", aggfunc="sum", fill_value=0)
    plot_stacked_bar(pvt_sum, "Saldo por SEGMENTO × CLASIFICACIÓN", "Segmento", "Saldo total")

    # Segmento × Subsegmento (Top 15 por saldo)
    g = (df.groupby(["segmento","subsegmento"])["saldo"]
            .sum().sort_values(ascending=False).head(15))
    idx_labels = [f"{seg} — {sub}" for (seg, sub) in g.index]
    plt.figure()
    plt.barh(idx_labels[::-1], g.values[::-1])  # mostrar mayor arriba
    plt.title("Top 15 SEGMENTO × SUBSEGMENTO por saldo")
    plt.xlabel("Saldo total")
    plt.ylabel("Segmento — Subsegmento")
    plt.tight_layout()

    # Clasificación × Situación (limitar a Top 8 situaciones por saldo)
    top_sit = head_index(df, "situacion", n=8, by="saldo")
    df_sit = df[df["situacion"].isin(top_sit)]
    pvt_cs_cnt = pd.pivot_table(df_sit, index="clasificacion", columns="situacion",
                                values="codigo_cuentahabiente", aggfunc="size", fill_value=0)
    plot_stacked_bar(pvt_cs_cnt, "Conteo por CLASIFICACIÓN × (Top 8) SITUACIÓN",
                     "Clasificación", "Conteo")

    pvt_cs_sum = pd.pivot_table(df_sit, index="clasificacion", columns="situacion",
                                values="saldo", aggfunc="sum", fill_value=0)
    plot_stacked_bar(pvt_cs_sum, "Saldo por CLASIFICACIÓN × (Top 8) SITUACIÓN",
                     "Clasificación", "Saldo total")

    # Clasificación × Tipo de deudor (conteo y saldo)
    pvt_ctd_cnt = pd.pivot_table(df, index="clasificacion", columns="tipo_deudor",
                                 values="codigo_cuentahabiente", aggfunc="size", fill_value=0)
    plot_stacked_bar(pvt_ctd_cnt, "Conteo por CLASIFICACIÓN × TIPO DE DEUDOR",
                     "Clasificación", "Conteo")

    pvt_ctd_sum = pd.pivot_table(df, index="clasificacion", columns="tipo_deudor",
                                 values="saldo", aggfunc="sum", fill_value=0)
    plot_stacked_bar(pvt_ctd_sum, "Saldo por CLASIFICACIÓN × TIPO DE DEUDOR",
                     "Clasificación", "Saldo total")

def top10_usuarios_cde(df):
    mask = df["clasificacion"].isin(["C","D","E"])
    top = (df[mask].groupby(["codigo_cuentahabiente","nombre_cuentahabiente"])["saldo"]
                .sum().sort_values(ascending=False).head(10).reset_index())
    etiquetas = top["nombre_cuentahabiente"].fillna("") + " (" + top["codigo_cuentahabiente"] + ")"
    plt.figure()
    plt.barh(etiquetas.iloc[::-1], top["saldo"].iloc[::-1].values)
    plt.title("Top 10 usuarios más endeudados (C/D/E)")
    plt.xlabel("Saldo total")
    plt.ylabel("Usuario")
    plt.tight_layout()

# ------------------ main ------------------
if __name__ == "__main__":
    df = leer_df(CSV_PATH)

    # Totales simples
    totales_simples(df)

    # Combinaciones solicitadas
    combinaciones(df)

    # Extra: top 10 C/D/E
    top10_usuarios_cde(df)

    # Mostrar todo
    plt.show()
