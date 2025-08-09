from pathlib import Path
import pandas as pd
from faker import Faker
import csv

# -------------------- Configuración ------------------------------------------
COLUMNAS = [
    "fecha_datos",
    "activo_crediticio",
    "tipo_activo",
    "codigo_deudor",
    "nombre_deudor",
    "tipo_persona",
    "segmento",
    "subsegmento",
    "dpi",
    "nit",
    "deudor_mayor",
    "cat_mayor",
    "cat_mayor_2",
    "alineacion",
]

TRADUCCION = {
    "1": "8",
    "2": "15",
    "3": "1",
    "4": "6",
    "5": "17",
    "6": "8",
    "7": "2",
    "8": "3",
    "9": "0",
}

faker = Faker(["en_US", "fr_FR", "de_DE", "it_IT", "pt_BR"])  # nombres extranjeros
# -----------------------------------------------------------------------------


def traducir_codigo(valor) -> str:
    if pd.isna(valor):
        return ""
    valor_str = str(valor).strip()
    return TRADUCCION.get(valor_str, valor_str)


def procesar_archivo(ruta_txt: Path) -> pd.DataFrame:
    df = pd.read_csv(
        ruta_txt,
        sep="|",
        header=None,
        names=COLUMNAS,
        dtype=str,
        engine="python",
        encoding="latin1",
        quoting=csv.QUOTE_NONE
    )

    # Limpieza
    df = df.astype(str).apply(lambda col: col.str.strip())

    # Traducción
    for campo in ("codigo_deudor", "dpi", "nit"):
        df[campo] = df[campo].apply(traducir_codigo)

    # Generar nombre único por código_deudor
    codigo_to_nombre = {}

    def obtener_nombre_ficticio(codigo):
        if codigo not in codigo_to_nombre:
            codigo_to_nombre[codigo] = faker.name()
        return codigo_to_nombre[codigo]

    df["nombre_deudor"] = df["codigo_deudor"].apply(obtener_nombre_ficticio)

    # Convertir fecha
    df["fecha_datos"] = pd.to_datetime(
        df["fecha_datos"], errors="coerce", format="%Y%m%d"
    )

    return df


def main():
    entrada = input("Nombre del archivo .txt (ej. data_dummy.txt): ").strip()
    salida = input("Nombre para el archivo .csv (ej. resultado.csv): ").strip()

    ruta_txt = Path(entrada)
    ruta_csv = Path(salida)

    df = procesar_archivo(ruta_txt)
    df.to_csv(ruta_csv, index=False, encoding="utf-8", sep=";")
    print(f"✅ CSV generado correctamente: {ruta_csv}")


if __name__ == "__main__":
    main()
