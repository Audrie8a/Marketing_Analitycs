import pandas as pd
import numpy as np
import random
from faker import Faker
from pathlib import Path
from collections import deque

# --- Catálogos ---
SEGMENTOS = {
    1: "Creditos empresariales",
    2: "Creditos productivos",
    3: "Creditos hipotecarios para vivienda y cédulas hipotecarias",
    4: "Creditos de consumo",
}
SUBSEGMENTOS = {
    11: "Comercio", 12: "Industrias manufactureras", 13: "Actividades Inmobiliarias y construcción",
    14: "Suministros de electricidad, gas y agua", 15: "Establecimientos financieros",
    16: "Agricultura, ganadería, silvicultura y pesca", 17: "Servicios y otros",
    21: "Comercio", 22: "Servicios y otros", 31: "Hipotecarios para vivienda",
    32: "Cédulas hipotecarias", 41: "Tarjetas de crédito", 42: "Vehículos",
    43: "Préstamos personales"
}
TIPO_DEUDOR = {
    1: "INDIVIDUAL NACIONAL", 2: "JURIDICA NACIONAL",
    3: "INDIVIDUAL EXTRANJERA", 4: "JURIDICA EXTRANJERA"
}
# Situaciones genéricas 1..9 + mapeos conocidos
SITUACIONES = {i: f"Situacion_{i}" for i in range(1, 10)}
SITUACIONES.update({
    15: "Tarjeta de Credito", 16: "Prestamos", 17: "Pagos Sobre Titulos de Capitalizacion",
    18: "No solicitado", 19: "Documento Descontados", 20: "Credito en Cuenta",
    21: "Arrendamiento Financiero", 98: "Otros Activos Crediticios relacionados",
    23: "Venta de Inmuebles y Muebles", 24: "Factoraje", 95: "Otros no relacionados",
    26: "Pagos por Carta de Credito", 27: "Venta de Activos Extraordinarios",
    28: "Documento por Cobrar", 29: "Cedulas hipotecarias", 30: "Tarjeta de crédito factorada"
})
CLASIF_SCORE = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}

faker = Faker(["es_ES", "en_US", "pt_BR"])


# ---------------------- Utilidades ----------------------
def parse_totales(cadena: str) -> dict:
    """Convierte 'A:10,B:5' o '1:20,2:3' -> dict con valores float."""
    out = {}
    for par in cadena.split(","):
        if ":" in par:
            k, v = par.split(":", 1)
            k = k.strip()
            try:
                out[k] = float(v)
            except Exception:
                out[k] = 0.0
    return out


def _scale_vector_to_target(counts_by_key: dict, keys_order: list, target: int) -> dict:
    """
    Escala un dict de conteos para que sume 'target' conservando proporciones
    (método de los mayores restos). Devuelve {key: int}.
    """
    base = np.array([float(counts_by_key.get(k, 0)) for k in keys_order], dtype=float)
    total = base.sum()
    if total <= 0 or target <= 0:
        return {k: 0 for k in keys_order}

    scaled = base * (target / total)
    floor_ = np.floor(scaled).astype(int)
    rem = int(target - floor_.sum())
    if rem > 0:
        frac = scaled - floor_
        idx = np.argsort(-frac)[:rem]
        floor_[idx] += 1
    return {k: int(v) for k, v in zip(keys_order, floor_)}


def _resolve_txt_inputs(path_txt: str | Path) -> list[Path]:
    """
    Acepta:
      - ruta a archivo .txt
      - carpeta con uno o más .txt
    Devuelve lista de archivos .txt a procesar. Si no encuentra, levanta error
    mostrando el contenido de la carpeta.
    """
    p = Path(path_txt)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent / p

    if p.is_dir():
        files = sorted(p.glob("*.txt"))
        if files:
            return files
        contenido = "\n".join(f"- {x.name}" for x in p.iterdir())
        raise FileNotFoundError(f"No se encontraron .txt en {p}\nContenido:\n{contenido}")

    if p.is_file():
        if p.suffix.lower() != ".txt":
            raise FileNotFoundError(f"Se esperaba .txt, recibí: {p}")
        return [p]

    raise FileNotFoundError(f"No existe la ruta: {p}")


# ---------------------- Generador principal ----------------------
def generar_dummy_desde_archivo(
    path_txt: str,
    salida_csv: str,
    max_registros: int | None = None,
    seed: int = 42,
    ratio_usuarios: float = 3.0,
    saldo_noise: bool = False,       # True -> reparte saldos con Dirichlet y conserva la suma
    dirichlet_alpha: float = 1.0,
) -> int:
    """
    Lee uno o varios .TXT (formato '|' con totales por clase/personería/situación) y genera filas dummy.

    - Si pides menos filas de las disponibles -> submuestreo proporcional.
    - Si pides más -> sobremuestreo proporcional (se avisa).
    - El saldo total se preserva proporcionalmente al factor de escala global.
    - Si 'saldo_noise' es True, reparte el saldo por clase con Dirichlet (misma suma, mayor variación).

    Retorna: total de filas escritas.
    """
    random.seed(seed); np.random.seed(seed)

    txt_files = _resolve_txt_inputs(path_txt)

    # Leer todas las líneas de todos los archivos (saltando el header de cada uno)
    lines = []
    for fp in txt_files:
        raw_lines = fp.read_text(encoding="latin1", errors="replace").splitlines()
        if not raw_lines:
            continue
        lines.extend(raw_lines[1:])  # saltar encabezado

    if not lines:
        return 0

    # ---------- PASO 1: pre-scan (cuánto hay disponible por línea) ----------
    registros = []
    total_disponible = 0

    for raw in lines:
        parts = [x.strip() for x in raw.split("|")]
        # FECHA | ID_SEGMENTO | ID_SUBSEGMENTO | COUNT_CLAS | SUMA_CLAS | COUNT_PERSONERIA | SUMA_SITUACION
        if len(parts) < 7:
            continue

        fecha_str = parts[0]
        try:
            mes_str = pd.to_datetime(fecha_str, dayfirst=True).strftime("%Y-%m")
        except Exception:
            mes_str = pd.to_datetime(fecha_str, errors="coerce").strftime("%Y-%m")

        id_segmento = int(float(parts[1]))
        id_subsegmento = int(float(parts[2]))

        clas_count = parse_totales(parts[3])  # A..E
        clas_suma  = parse_totales(parts[4])  # A..E
        tipo_count = parse_totales(parts[5])  # 1..4
        situ_suma  = parse_totales(parts[6])  # 1..9

        n_line = int(sum(clas_count.get(k, 0) for k in CLASIF_SCORE.keys()))
        if n_line <= 0:
            continue

        registros.append({
            "mes": mes_str,
            "id_segmento": id_segmento,
            "id_subsegmento": id_subsegmento,
            "clas_count": {k: float(clas_count.get(k, 0)) for k in CLASIF_SCORE.keys()},
            "clas_suma":  {k: float(clas_suma.get(k, 0))  for k in CLASIF_SCORE.keys()},
            "tipo_count": {int(k): float(v) for k, v in tipo_count.items() if str(k).isdigit()},
            "situ_pos":   [int(k) for k, v in situ_suma.items() if str(k) != "NAN" and float(v) > 0],
            "n_line": n_line
        })
        total_disponible += n_line

    if total_disponible == 0:
        return 0

    objetivo = max_registros if (max_registros and max_registros > 0) else total_disponible
    factor = objetivo / total_disponible

    if objetivo > total_disponible:
        print(f"[AVISO] Pediste {objetivo:,} filas y solo hay {total_disponible:,}. "
              f"Se hará sobremuestreo proporcional (factor {factor:.4f}).")

    wrote_header = False
    total_escritos = 0
    carry = 0.0  # para redondeo por línea
    nombre_cache: dict[str, str] = {}

    # ---------- PASO 2: generación proporcional por línea ----------
    for rec in registros:
        # filas a generar para esta línea (proporcional + acarreo de redondeo)
        esperado = rec["n_line"] * factor + carry
        n_generar = int(round(esperado))
        carry = esperado - n_generar

        # no pasarse del objetivo total por acumulación de redondeos
        if total_escritos + n_generar > objetivo:
            n_generar = objetivo - total_escritos
        if n_generar <= 0:
            continue

        # --- repartir por CLASE respetando proporciones originales ---
        clases = list(CLASIF_SCORE.keys())  # ["A","B","C","D","E"]
        gen_por_clase = _scale_vector_to_target(rec["clas_count"], clases, n_generar)

        # --- usuarios/nombres ---
        max_usuarios = max(1000, int(n_generar / ratio_usuarios))
        usuarios = [f"U{str(i).zfill(6)}" for i in range(1, max_usuarios + 1)]
        random.shuffle(usuarios)
        usuarios_rep = random.choices(usuarios, k=n_generar)
        for u in set(usuarios_rep):
            nombre_cache.setdefault(u, faker.name())

        # --- tipos deudor proporcional ---
        tipos_keys = sorted(set(int(k) for k in rec["tipo_count"].keys()) or {1, 2, 3, 4})
        gen_por_tipo = _scale_vector_to_target(rec["tipo_count"], tipos_keys, n_generar)
        tipos_pool = []
        for k in tipos_keys:
            tipos_pool.extend([int(k)] * int(gen_por_tipo.get(k, 0)))
        if len(tipos_pool) < n_generar:
            tipos_pool.extend(random.choices(tipos_keys, k=n_generar - len(tipos_pool)))
        random.shuffle(tipos_pool)
        tipos_queue = deque(tipos_pool)

        # --- situaciones disponibles ---
        situaciones_disp = rec["situ_pos"] or [1]

        # --- saldo objetivo por clase (promedio original * filas generadas) ---
        saldo_obj_por_clase = {}
        for c in clases:
            cnt_orig = rec["clas_count"].get(c, 0.0)
            suma_orig = rec["clas_suma"].get(c, 0.0)
            cnt_gen = gen_por_clase.get(c, 0)
            if cnt_orig > 0 and suma_orig > 0 and cnt_gen > 0:
                total_c = (suma_orig / cnt_orig) * cnt_gen  # preserva suma proporcional
            else:
                total_c = 0.0
            saldo_obj_por_clase[c] = total_c

        # --- construir filas ---
        rows = []
        idx_usuario = 0
        for c in clases:
            cnt_c = gen_por_clase.get(c, 0)
            if cnt_c <= 0:
                continue
            score = CLASIF_SCORE[c]

            if saldo_noise and saldo_obj_por_clase[c] > 0:
                # reparte con Dirichlet (misma suma, variación en valores)
                w = np.random.dirichlet([dirichlet_alpha] * cnt_c)
                saldos = (w * saldo_obj_por_clase[c]).tolist()
            else:
                prom = saldo_obj_por_clase[c] / cnt_c if cnt_c else 0.0
                saldos = [prom] * cnt_c

            for saldo in saldos:
                usuario = usuarios_rep[idx_usuario]; idx_usuario += 1
                tipo_sel = tipos_queue.popleft() if tipos_queue else random.choice(tipos_keys)
                situ_sel = random.choice(situaciones_disp)

                rows.append([
                    rec["mes"], usuario, nombre_cache[usuario],
                    rec["id_segmento"], SEGMENTOS.get(rec["id_segmento"], f"SEG-{rec['id_segmento']}"),
                    rec["id_subsegmento"], SUBSEGMENTOS.get(rec["id_subsegmento"], f"SUB-{rec['id_subsegmento']}"),
                    int(tipo_sel), TIPO_DEUDOR.get(int(tipo_sel), f"TIPO-{tipo_sel}"),
                    int(situ_sel), SITUACIONES.get(int(situ_sel), f"SIT-{situ_sel}"),
                    c, score, float(saldo)
                ])

        if rows:
            df_chunk = pd.DataFrame(rows, columns=[
                "mes", "codigo_cuentahabiente", "nombre_cuentahabiente",
                "id_segmento", "segmento",
                "id_subsegmento", "subsegmento",
                "id_tipo_deudor", "tipo_deudor",
                "id_situacion", "situacion",
                "clasificacion", "score", "saldo"
            ])
            df_chunk.to_csv(salida_csv, mode="a", index=False, header=not wrote_header, encoding="utf-8")
            wrote_header = True
            total_escritos += len(df_chunk)

        if total_escritos >= objetivo:
            break

    return total_escritos

def diagnostico_capacidad(path_txt: str | Path, max_registros: int | None = None, mostrar=True):
    """
    Lee el/los TXT y reporta:
      - total disponible (suma de A..E)
      - objetivo (max_registros o todo)
      - factor (objetivo/disponible)
      - modo: submuestreo / sobremuestreo / exacto
      - desglose por segmento y por clase
    """
    txt_files = _resolve_txt_inputs(path_txt)

    total_disponible = 0
    por_segmento: dict[int, int] = {}
    por_clase = {k: 0 for k in CLASIF_SCORE.keys()}  # A..E

    for fp in txt_files:
        lines = fp.read_text(encoding="latin1", errors="replace").splitlines()
        for raw in lines[1:]:
            parts = [x.strip() for x in raw.split("|")]
            if len(parts) < 7:
                continue
            seg = int(float(parts[1]))
            clas_count = parse_totales(parts[3])  # A..E, NAN posible

            n_line = int(sum(clas_count.get(k, 0) for k in CLASIF_SCORE.keys()))
            if n_line <= 0:
                continue

            total_disponible += n_line
            por_segmento[seg] = por_segmento.get(seg, 0) + n_line
            for k in CLASIF_SCORE.keys():
                por_clase[k] += int(clas_count.get(k, 0))

    objetivo = total_disponible if not max_registros or max_registros <= 0 else int(max_registros)
    factor = objetivo / total_disponible if total_disponible else 0.0
    if objetivo < total_disponible:
        modo = "submuestreo"
    elif objetivo > total_disponible:
        modo = "sobremuestreo"
    else:
        modo = "exacto"

    resumen = {
        "archivos": [str(p) for p in txt_files],
        "total_disponible": int(total_disponible),
        "objetivo": int(objetivo),
        "factor": float(factor),
        "modo": modo,
        "por_segmento": dict(sorted(por_segmento.items())),
        "por_clase": por_clase,
    }

    if mostrar:
        print("=== Diagnóstico de capacidad ===")
        print(f"Archivos: {len(txt_files)}")
        print(f"Total disponible (A..E): {total_disponible:,}")
        print(f"Objetivo (max_registros): {objetivo:,}")
        print(f"Factor (objetivo/disponible): {factor:.4f}  -> {modo.upper()}")
        print("\nPor segmento (registros disponibles):")
        for s, n in sorted(por_segmento.items()):
            print(f"  - Segmento {s}: {n:,}")
        print("\nPor clase (disponible):")
        for k in ["A","B","C","D","E"]:
            print(f"  {k}: {por_clase[k]:,}")

        if modo == "sobremuestreo":
            print("\n[AVISO] Estás pidiendo más de lo disponible. Se generará sobremuestreo proporcional.")

    return resumen

# ---------------------- Ejecutable ----------------------
if __name__ == "__main__":
    base = Path(__file__).resolve().parent

    # Candidatos más comunes (ajusta nombres si difieren)
    candidates = [
        base / "Data_Proyecto" / "JAN25",            # carpeta con varios .txt
        base / "Data_Proyecto" / "JAN25.txt",        # archivo dentro de Data_Proyecto
        base / "JAN25.txt",                          # archivo en la raíz del proyecto
    ]

    # Escoge el primero que exista
    path_txt = next((p for p in candidates if p.exists()), None)
    if path_txt is None:
        raise FileNotFoundError(
            "No encontré ninguna de estas rutas:\n" +
            "\n".join(f"- {p}" for p in candidates)
        )
    
    diag = diagnostico_capacidad(path_txt, max_registros=400_000)

    # Carpeta de salida y archivo
    out_dir = base / "Data_Dummy"
    out_dir.mkdir(parents=True, exist_ok=True)
    salida_csv = out_dir / "JAN25.csv"
    if salida_csv.exists():
        salida_csv.unlink()  # limpiar para no acumular

    # Diagnóstico útil
    p = Path(path_txt)
    print(f"Fuente: {p}")
    print("exists:", p.exists(), "is_dir:", p.is_dir(), "is_file:", p.is_file())
    if p.is_dir():
        # OJO: patrón correcto es *.txt o **/*.txt (recursivo)
        encontrados = [x.name for x in p.glob("*.txt")]
        print("TXT encontrados:", encontrados if encontrados else "(ninguno)")
        

    total = generar_dummy_desde_archivo(
        path_txt=str(path_txt),
        salida_csv=str(salida_csv),
        max_registros=400_000,   # None para todo lo disponible
        seed=42,
        ratio_usuarios=3.0,
        saldo_noise=False,       # True para variar saldos conservando la suma
        dirichlet_alpha=1.0
    )
    print(f"Filas escritas: {total:,}")
