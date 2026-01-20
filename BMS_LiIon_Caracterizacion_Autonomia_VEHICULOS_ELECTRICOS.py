#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
# PROYECTO:
#   "Caracterización de un sistema de administración de energía (BMS) basado en
#    baterías Li-ion para maximizar la autonomía de vehículos eléctricos"
#
# ARCHIVO / MÓDULO:
#   BMS_LiIon_Caracterizacion_Autonomia_Vehiculos_Electricos.py
#
# FINALIDAD DEL DOCUMENTO (PARA IMPI):
#   Este archivo constituye el código fuente completo a registrar ante el IMPI.
#   Incluye:
#       - Descripción detallada de cada componente (clases, funciones y flujos).
#       - Contratos de datos (entradas/salidas en CSV, flujo serie y objetos).
#       - Justificación técnica de diseño (modelos, algoritmos y métricas).
#       - Casos de uso y ejemplos de ejecución.
#   El código es original de los autores listados en este encabezado.
#
# FUNCIONALIDAD DEL SOFTWARE:
#   1) Simulación de un pack LiFePO4 48 V – 20 Ah (16S) mediante un modelo de Thévenin:
#        - Dinámica eléctrica mediante R0 y la rama R1–C1.
#        - Curva OCV–SOC para estimación de tensión en reposo.
#        - Escenarios de conducción urbano y extraurbano.
#   2) Estimación híbrida del SOC:
#        - Conteo de Coulomb.
#        - Estimación por tensión (OCV inversa).
#        - Fusión mediante filtro complementario (ponderación alpha).
#   3) Ingesta de datos reales:
#        - Desde archivos CSV con columnas t_ms/t_s, V, I y P.
#        - Desde puerto serie en formato: "t_ms,V,I,P".
#   4) Cálculo de KPIs:
#        - Energía neta, pérdidas resistivas (I²R0), Vmin/Vmax.
#        - MAE y RMSE del SOC (cuando existe la referencia SOC_true).
#   5) Exportación automática:
#        - Archivos CSV, JSON, Markdown y gráficos PNG.
#   6) Subida opcional a Firebase (Storage + Firestore).
#
# CONTRATOS DE DATOS (DATA CONTRACTS):
#   - CSV (ingesta):
#         t_ms o t_s, V, I, P (opcional), E_Wh (si falta, se calcula).
#   - Serie:
#         "t_ms,V,I,P"
#   - Simulación:
#         DataFrame con t_s, V, I, P, E_Wh, SOC_true y SOC_est.
#
# DEPENDENCIAS OPCIONALES:
#   numpy, pandas, matplotlib, tabulate, pyserial, firebase-admin
#
# EJEMPLOS DE EJECUCIÓN:
#   # 1) Simulación completa:
#       python3 BMS_LiIon_Caracterizacion_Autonomia_Vehiculos_Electricos.py --mode simulate
#
#   # 2) Procesar CSV real:
#       python3 BMS_LiIon_Caracterizacion_Autonomia_Vehiculos_Electricos.py --mode ingest-csv --csv archivo.csv
#
#   # 3) Capturar por puerto serie 120 s:
#       python3 BMS_LiIon_Caracterizacion_Autonomia_Vehiculos_Electricos.py --mode ingest-serial --duration 120
#
#   # 4) Subida a Firebase:
#       python3 BMS_LiIon_Caracterizacion_Autonomia_Vehiculos_Electricos.py --mode simulate --firebase \
#               --fb-creds cred.json --fb-bucket mi-bucket
#
# AUTORÍA:
#   M.C. Héctor Javier Jarquín-Flores
#   M.C. Álvaro César Guevara-Ramírez
#   Dr. Carlos Mauricio Lastre-Domínguez
#   Dr. Eric Mario Silva-Cruz
#
# AFILIACIÓN:
#   Tecnológico Nacional de México / Instituto Tecnológico de Oaxaca
#
# FECHA:
#   Creación: 2025-02-07
#   Última modificación: 2025-02-07
#
# VERSIÓN:
#   1.1 (documentación ampliada para registro ante IMPI)
#
# LICENCIA:
#   Reservados todos los derechos por los autores. Uso académico y de investigación.
# =============================================================================
from __future__ import annotations

# --------------------------- 1. IMPORTACIONES BÁSICAS ------------------------
# Descripción: Librerías estándar y opcionales. El software debe ser funcional
# incluso si algunas dependencias opcionales no están instaladas (degrada
# elegantemente funciones no críticas).
import os
import sys
import csv
import time
import json
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# Opcionales numéricos/analíticos
try:
    import numpy as np
except Exception:
    np = None  # El módulo sim requiere numpy/pandas. Se valida en tiempo de ejecución.

try:
    import pandas as pd
except Exception:
    pd = None  # Sin pandas, sólo algunas utilidades mínimas estarán disponibles.

# Opcional para gráficas
try:
    import matplotlib.pyplot as plt
    _HAVE_PLOT = True
except Exception:
    _HAVE_PLOT = False

# Opcional tabla MD bonita
try:
    from tabulate import tabulate
    _HAVE_TABULATE = True
except Exception:
    _HAVE_TABULATE = False

# Opcional Serie
try:
    import serial  # pyserial
except Exception:
    serial = None  # La ingesta por serie no estará disponible si falta.

# Opcional Firebase
try:
    import firebase_admin
    from firebase_admin import credentials, firestore, storage
    _HAVE_FIREBASE = True
except Exception:
    _HAVE_FIREBASE = False


# --------------------------- 2. CONFIGURACIÓN GLOBAL -------------------------
@dataclass
class Config:
    """
    Config: parámetros de ejecución, rutas y banderas.
    - out_dir: carpeta donde se escriben resultados (CSV/PNG/JSON/MD).
    - simulate: si True, corre la suite de simulación (urbano/extraurbano).
    - ingest_csv: ruta a CSV a procesar (modo ingest-csv).
    - ingest_serial: si True, captura desde puerto serie (modo ingest-serial).
    - serial_port/baud/timeout: parámetros de comunicación serie.
    - sample_dt: paso temporal (s) para simulación/estimación (por defecto 1.0 s).
    - firebase_*: parámetros opcionales para subir resultados a Firebase.
    - project_tag: identificador lógico para nombrar recursos en la nube.
    """
    out_dir: Path = Path("out")
    simulate: bool = True
    ingest_csv: Optional[Path] = None
    ingest_serial: bool = False
    serial_port: str = "/dev/ttyACM0"
    serial_baud: int = 115200
    serial_timeout: float = 2.0
    sample_dt: float = 1.0
    firebase_enable: bool = False
    firebase_creds_json: Optional[Path] = None
    firebase_bucket: Optional[str] = None
    project_tag: str = "BMS_LFP_48V20Ah"


# --------------------------- 3. UTILIDADES DE ARCHIVO ------------------------
def ensure_out_dir(cfg: Config) -> None:
    """
    Crea el directorio de salida si no existe.
    Justificación: mantener trazabilidad y no fallar por rutas inexistentes.
    """
    cfg.out_dir.mkdir(parents=True, exist_ok=True)


def save_json(obj: dict, path: Path) -> None:
    """
    Guarda un dict en JSON legible (UTF-8) para auditoría/reporte.
    Entrada:
      - obj: diccionario serializable
      - path: ruta de salida
    Salida: archivo JSON en disco.
    """
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(rows: List[dict], fieldnames: List[str], path: Path) -> None:
    """
    Escribe un CSV a partir de una lista de dicts con 'fieldnames' fijos.
    Garantiza encabezado y orden de columnas.
    """
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


# --------------------------- 4. CONECTORES A FIREBASE ------------------------
def firebase_init(cfg: Config):
    """
    Inicializa Firebase si está habilitado y disponible.
    Requisitos:
      - firebase_admin instalado
      - credenciales JSON válidas y bucket configurado
    Retorna:
      - app Firebase o None si no procede.
    """
    if not cfg.firebase_enable:
        print("[Firebase] Deshabilitado (bandera --firebase no activa).")
        return None
    if not _HAVE_FIREBASE:
        print("[Firebase] Paquete 'firebase_admin' no instalado.")
        return None
    if not cfg.firebase_creds_json or not cfg.firebase_creds_json.exists():
        print("[Firebase] Falta archivo de credenciales JSON.")
        return None
    cred = credentials.Certificate(str(cfg.firebase_creds_json))
    app = firebase_admin.initialize_app(cred, {
        "storageBucket": cfg.firebase_bucket
    }) if not firebase_admin._apps else firebase_admin.get_app()
    print("[Firebase] Inicializado correctamente.")
    return app


def firebase_upload_file(local_path: Path, remote_name: str) -> None:
    """
    Sube un archivo local al bucket de Firebase Storage.
    Entradas:
      - local_path: ruta en disco
      - remote_name: nombre de objeto remoto (carpeta lógica/archivo)
    """
    if not _HAVE_FIREBASE:
        return
    try:
        bucket = storage.bucket()
        blob = bucket.blob(remote_name)
        blob.upload_from_filename(str(local_path))
        print(f"[Firebase] Subido: {remote_name}")
    except Exception as e:
        print(f"[Firebase] Error subiendo {remote_name}: {e}")


def firebase_upload_doc(collection: str, doc_id: str, data: dict) -> None:
    """
    Crea/actualiza un documento en Firestore.
    Útil para mantener un registro ligero (metadatos, KPIs resumidos).
    """
    if not _HAVE_FIREBASE:
        return
    try:
        db = firestore.client()
        db.collection(collection).document(doc_id).set(data, merge=True)
        print(f"[Firebase] Doc {collection}/{doc_id} actualizado.")
    except Exception as e:
        print(f"[Firebase] Error Firestore: {e}")


# --------------------------- 5. MODELO DE BATERÍA (LFP) ----------------------
# Curva OCV por celda (V) vs SOC (%) para LFP (aprox. conservadora).
# Importante: reemplazar por curva del fabricante si se dispone.
OCV_TABLE = [
    (100, 3.395), (95, 3.380), (90, 3.370), (80, 3.360),
    (70, 3.345),  (60, 3.330), (50, 3.320), (40, 3.315),
    (30, 3.305),  (20, 3.290), (10, 3.250), ( 0, 3.000),
]


def ocv_cell_from_soc(soc: float) -> float:
    """
    Calcula la OCV (V) por celda a partir de SOC(%) con interpolación lineal.

    Entradas:
      - soc: State of Charge en porcentaje [0..100]

    Proceso:
      - Se recorta al rango [0,100].
      - Se busca el intervalo en OCV_TABLE tal que s2 <= SOC <= s1
      - Interpola entre (s1,v1) y (s2,v2).

    Salida:
      - OCV de la celda en volts (float).
    """
    s = max(0.0, min(100.0, float(soc)))
    for (s1, v1), (s2, v2) in zip(OCV_TABLE[:-1], OCV_TABLE[1:]):
        if s2 <= s <= s1:
            a = (v2 - v1) / (s2 - s1)
            return v1 + a * (s - s1)
    return OCV_TABLE[-1][1] if s <= 0 else OCV_TABLE[0][1]


def soc_from_pack_voltage(v_pack: float, n_series: int = 16) -> float:
    """
    Estima SOC(%) a partir de la tensión del pack (V_pack) invirtiendo la OCV.
    Aproximación por tramos lineales (válida para condiciones cercanas a reposo).

    Entradas:
      - v_pack : tensión del pack en V
      - n_series: número de celdas en serie (16 para 48V LFP típico)

    Salida:
      - SOC (%) en [0..100]
    """
    v_cell = float(v_pack) / float(n_series)
    for (s1, v1), (s2, v2) in zip(OCV_TABLE[:-1], OCV_TABLE[1:]):
        v_hi, v_lo = v1, v2
        s_hi, s_lo = s1, s2
        if v_lo <= v_cell <= v_hi:
            return s_hi + (s_lo - s_hi) * (v_cell - v_hi) / (v_lo - v_hi)
    return 0.0 if v_cell < OCV_TABLE[-1][1] else 100.0


@dataclass
class LFPTheveninPack:
    """
    Modelo de Thévenin (un polo) para un pack LFP 48 V–20 Ah (16S).

    Parámetros:
      - Q_Ah   : capacidad nominal (Ah)
      - N_series: número de celdas en serie (16 -> ~51.2V nominal)
      - R0     : resistencia serie equivalente del pack (ohm)
      - R1, C1 : rama de polarización (captura dinámica de sobre/polarización)
      - soc0   : estado de carga inicial (%)
      - vrc0   : tensión inicial en la rama RC (V)

    Método step(I, dt, state):
      - Integra un paso de tiempo (Euler):
          vrc(t+dt) = vrc + (-vrc/(R1*C1) + I/C1)*dt
          v_oc      = N_series * OCV_cell(SOC)
          v_term    = v_oc - I*R0 - vrc
          SOC(t+dt) = SOC - (I*dt)/(Q_Ah*3600)*100
      - Devuelve ((soc, vrc), v_term)
    """
    Q_Ah: float = 20.0
    N_series: int = 16
    R0: float = 0.050
    R1: float = 0.020
    C1: float = 1000.0
    soc0: float = 100.0
    vrc0: float = 0.0

    def step(self, I: float, dt: float, state: Tuple[float, float]) -> Tuple[Tuple[float, float], float]:
        soc, vrc = state
        vrc += (-vrc / (self.R1 * self.C1) + I / self.C1) * dt
        v_oc = self.N_series * ocv_cell_from_soc(soc)
        v_term = v_oc - I * self.R0 - vrc
        soc -= (I * dt) / (self.Q_Ah * 3600.0) * 100.0
        soc = max(0.0, min(100.0, soc))
        return (soc, vrc), v_term


# --------------------------- 6. ESCENARIOS DE CORRIENTE ----------------------
def scenario_urbano() -> List[Tuple[int, float]]:
    """
    Escenario de demanda "urbano".
    Retorna una lista de tuplas (duración_en_segundos, corriente_A) donde
    corriente > 0 indica tracción y corriente < 0 regeneración.
    """
    return [
        (30, 10), (15, 25), (10,  0), (20, -8),
        (40, 18), (15, 30), (20,  5), (10, -12),
        (30, 15), (20,  0),
    ]


def scenario_extraurbano() -> List[Tuple[int, float]]:
    """
    Escenario de demanda "extraurbano" (más sostenido).
    """
    return [
        (60, 15), (40, 35), (30, 10), (25, -15),
        (80, 30), (20,  5), (30, -10), (90,  28),
        (20,  0),
    ]


# --------------------------- 7. SIMULACIÓN (núcleo) --------------------------
def simulate(pack: LFPTheveninPack, schedule: List[Tuple[int, float]], dt: float = 1.0) -> 'pd.DataFrame':
    """
    Ejecuta una simulación completo a partir de un schedule (escenario).
    Salida (DataFrame):
      - t_s    : tiempo (s)
      - V      : tensión pack (V)
      - I      : corriente (A)
      - P      : potencia (W) = V*I
      - E_Wh   : energía acumulada (Wh)
      - SOC_true: SOC (%) de referencia interna del simulador

    Restricción: requiere pandas/numpy. Se valida antes de ejecuciones de suite.
    """
    if pd is None:
        raise RuntimeError("pandas no está instalado. Instala con: pip install pandas")

    t = 0.0
    state = (pack.soc0, pack.vrc0)
    rows = []
    e_wh = 0.0
    for dur, I in schedule:
        steps = int(round(dur / dt))
        for _ in range(steps):
            state, v = pack.step(I, dt, state)
            soc_true, _ = state
            p = v * I
            e_wh += p * dt / 3600.0
            rows.append((t, v, I, p, e_wh, soc_true))
            t += dt
    df = pd.DataFrame(rows, columns=["t_s", "V", "I", "P", "E_Wh", "SOC_true"])
    return df


# --------------------------- 8. ESTIMACIÓN DE SOC ---------------------------
def soc_hybrid_df(df: 'pd.DataFrame', Q_Ah=20.0, dt=1.0, alpha=0.85, soc0=100.0, n_series=16) -> 'pd.DataFrame':
    """
    Agrega columna 'SOC_est' al DataFrame fusionando:
      - SOC por Coulomb: SOC_c = SOC_prev - (I*dt)/(Q_Ah*3600)*100
      - SOC por OCV    : SOC_v = inv_OCV(V_pack)
      - Fusión         : SOC   = alpha*SOC_c + (1-alpha)*SOC_v

    Entradas:
      - df     : DataFrame con columnas V, I, t_s (E_Wh opcional)
      - Q_Ah   : capacidad nominal (Ah)
      - dt     : paso temporal (s). Si el df no tiene paso constante, se recomienda
                 calcularlo como promedio de diff(t_s).
      - alpha  : peso del conteo de Coulomb (0..1)
      - soc0   : SOC inicial asumido (%)
      - n_series: # celdas en serie para inv. OCV

    Salida:
      - DataFrame copia con columna adicional 'SOC_est'.
    """
    if pd is None:
        raise RuntimeError("pandas no está instalado. Instala con: pip install pandas")

    soc = soc0
    out = []
    for v, I in zip(df["V"].to_numpy(), df["I"].to_numpy()):
        soc_c = soc - (I * dt) / (Q_Ah * 3600.0) * 100.0
        soc_v = soc_from_pack_voltage(v, n_series=n_series)
        soc = alpha * soc_c + (1 - alpha) * soc_v
        soc = float(max(0.0, min(100.0, soc)))
        out.append(soc)
    df2 = df.copy()
    df2["SOC_est"] = out
    return df2


# --------------------------- 9. KPIs Y MÉTRICAS ------------------------------
def compute_kpis(df: 'pd.DataFrame', pack: LFPTheveninPack) -> Dict[str, float]:
    """
    Calcula indicadores clave:
      - E_Wh_net: energía neta al final (Wh).
      - Vmin/Vmax: tensiones extremas (V).
      - Loss_Wh : pérdidas resistivas ~ sum(I^2 * R0) * dt / 3600 (Wh).
      - Si hay SOC_true y SOC_est:
          SOC_final_true, SOC_final_est, SOC_MAE_% y SOC_RMSE_%

    Retorna:
      dict de KPIs con floats.
    """
    kpis = {
        "E_Wh_net": float(df["E_Wh"].iloc[-1]) if "E_Wh" in df else float((df["V"] * df["I"]).sum() * (df["t_s"].diff().fillna(0.0).mean() or 1.0) / 3600.0),
        "Vmin":     float(df["V"].min()),
        "Vmax":     float(df["V"].max()),
    }
    if "SOC_true" in df and "SOC_est" in df:
        kpis["SOC_final_true"] = float(df["SOC_true"].iloc[-1])
        kpis["SOC_final_est"]  = float(df["SOC_est"].iloc[-1])
        diff = (df["SOC_est"] - df["SOC_true"])
        kpis["SOC_MAE_%"] = float(diff.abs().mean())
        kpis["SOC_RMSE_%"] = float((diff.pow(2).mean()) ** 0.5)

    dt = float(df["t_s"].diff().fillna(0.0).mean() or 1.0)
    loss_wh = float((df["I"] ** 2 * pack.R0).sum() * dt / 3600.0)
    kpis["Loss_Wh"] = loss_wh
    return kpis


# --------------------------- 10. GRÁFICAS (opcional) -------------------------
def plot_vi(df: 'pd.DataFrame', title: str, out_png: Path) -> None:
    """
    Genera gráfico de Tensión (V) y Corriente (I) vs. tiempo.
    Salida: archivo PNG en 'out_png'.
    """
    if not _HAVE_PLOT:
        print(f"[Plot] matplotlib no disponible: {out_png.name} no se generó.")
        return
    fig, ax1 = plt.subplots(figsize=(7, 3))
    ax1.plot(df["t_s"], df["V"], label="V (V)")
    ax1.set_xlabel("Tiempo (s)")
    ax1.set_ylabel("Tensión (V)")
    ax2 = ax1.twinx()
    ax2.plot(df["t_s"], df["I"], label="I (A)", linestyle="--")
    ax2.set_ylabel("Corriente (A)")
    fig.tight_layout()
    fig.suptitle(title, y=1.02, fontsize=10)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_soc(df: 'pd.DataFrame', title: str, out_png: Path) -> None:
    """
    Genera gráfico de SOC verdadero (si existe) y estimado.
    """
    if not _HAVE_PLOT:
        print(f"[Plot] matplotlib no disponible: {out_png.name} no se generó.")
        return
    fig, ax = plt.subplots(figsize=(7, 3))
    if "SOC_true" in df:
        ax.plot(df["t_s"], df["SOC_true"], label="SOC verdadero")
    if "SOC_est" in df:
        ax.plot(df["t_s"], df["SOC_est"], label="SOC estimado", linestyle="--")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("SOC (%)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.suptitle(title, y=1.02, fontsize=10)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_soc_error(df: 'pd.DataFrame', title: str, out_png: Path) -> None:
    """
    Grafica el error (SOC_est - SOC_true) vs. tiempo si ambas series existen.
    Útil para visualizar sesgo/deriva del estimador.
    """
    if not _HAVE_PLOT:
        print(f"[Plot] matplotlib no disponible: {out_png.name} no se generó.")
        return
    if not {"SOC_true", "SOC_est"}.issubset(df.columns):
        print(f"[Plot] Columnas SOC_true/SOC_est no encontradas para error: {out_png.name}")
        return
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(df["t_s"], df["SOC_est"] - df["SOC_true"])
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Error SOC (p.p.)")
    fig.tight_layout()
    fig.suptitle(title, y=1.02, fontsize=10)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_kpis_table_md(kpis_rows: List[Dict[str, float]], out_md: Path) -> None:
    """
    Guarda KPIs en formato Markdown (tabla) si 'tabulate' está instalado.
    En caso contrario, guarda JSON simple como respaldo.
    """
    if _HAVE_TABULATE:
        cols = ["Escenario", "E_Wh_net", "Loss_Wh", "Vmin", "Vmax", "SOC_final_true", "SOC_final_est", "SOC_MAE_%", "SOC_RMSE_%"]
        cols = [c for c in cols if any(c in r for r in kpis_rows)]
        table = tabulate([[r.get(c, "") for c in cols] for r in kpis_rows], headers=cols, tablefmt="github", floatfmt=".3f")
        out_md.write_text(table, encoding="utf-8")
    else:
        out_md.write_text(json.dumps(kpis_rows, indent=2, ensure_ascii=False), encoding="utf-8")


# --------------------------- 11. INGESTA DE DATOS ----------------------------
def ingest_csv(path: Path) -> 'pd.DataFrame':
    """
    Lee un CSV del banco de pruebas y retorna un DataFrame estandarizado.

    Contrato:
      - Entradas mínimas:
          * t_ms (milisegundos)  O  t_s (segundos)
          * V, I
        P es opcional (si falta se calcula V*I).
      - Salida:
          DataFrame con t_s, V, I, P y E_Wh (Wh) acumulada.

    Mecanismo:
      - Si llega t_ms y no t_s, convierte t_ms -> t_s.
      - Si no existe E_Wh, integra potencia para construirla.
    """
    if pd is None:
        raise RuntimeError("pandas no está instalado. Instala con: pip install pandas")
    df = pd.read_csv(path)
    if "t_s" not in df and "t_ms" in df:
        df["t_s"] = df["t_ms"] * 1e-3
    if "P" not in df:
        df["P"] = df["V"] * df["I"]
    if "E_Wh" not in df:
        dt = float(df["t_s"].diff().fillna(0.0).mean() or 1.0)
        df["E_Wh"] = (df["P"] * dt / 3600.0).cumsum()
    return df


def ingest_serial_to_csv(cfg: Config, out_csv: Path, duration_s: Optional[int] = None) -> None:
    """
    Captura líneas por Serie y guarda un CSV estándar con:
      t_ms, V, I, P, t_s, E_Wh

    Formato de línea esperado:
      "t_ms,V,I,P"  (ej. "12345,12.72,1.18,15.00")

    - duration_s: si se especifica, detiene la captura tras ese tiempo.
    - La función NO estima SOC: sólo captura y deja el CSV listo para procesar.

    Requiere 'pyserial'.
    """
    if serial is None:
        raise RuntimeError("pyserial no está instalado. pip install pyserial")

    ser = serial.Serial(cfg.serial_port, cfg.serial_baud, timeout=cfg.serial_timeout)
    print(f"[Serie] Escuchando {cfg.serial_port} @ {cfg.serial_baud} ...")
    t0 = time.time()
    rows = []
    last_t_s = 0.0
    try:
        while True:
            line = ser.readline().decode(errors="ignore").strip()
            if not line:
                if duration_s and (time.time() - t0) > duration_s:
                    break
                continue
            try:
                t_ms_str, V_str, I_str, P_str = line.split(",")
                t_ms = int(float(t_ms_str))
                V = float(V_str); I = float(I_str); P = float(P_str)
                t_s = t_ms / 1000.0
                dt = max(0.0, t_s - last_t_s)
                last_t_s = t_s
                e_wh = (P * dt / 3600.0) + (rows[-1]["E_Wh"] if rows else 0.0)
                rows.append({"t_ms": t_ms, "V": V, "I": I, "P": P, "t_s": t_s, "E_Wh": e_wh})
            except Exception:
                # Ignora líneas corruptas o parciales.
                pass

            if duration_s and (time.time() - t0) > duration_s:
                break
    finally:
        ser.close()

    fieldnames = ["t_ms", "V", "I", "P", "t_s", "E_Wh"]
    write_csv(rows, fieldnames, out_csv)
    print(f"[Serie] {len(rows)} muestras guardadas en: {out_csv}")


# --------------------------- 12. FLUJOS "MAIN" -------------------------------
def run_simulation_suite(cfg: Config) -> None:
    """
    Ejecuta la suite de simulaciones (urbano y extraurbano) de punta a punta:
      - Simula -> estima SOC -> guarda CSV -> KPIs -> gráficas -> exporta resúmenes
      - Opcional: sube a Firebase (si bandera activa y credenciales válidas).
    """
    ensure_out_dir(cfg)
    if pd is None or np is None:
        raise RuntimeError("Se requieren pandas y numpy para la simulación. pip install numpy pandas")

    pack = LFPTheveninPack()
    scenarios = {
        "urbano": scenario_urbano(),
        "extraurbano": scenario_extraurbano(),
    }

    kpis_rows = []
    for name, sched in scenarios.items():
        # 1) Simulación
        df = simulate(pack, sched, dt=cfg.sample_dt)
        # 2) Estimación SOC híbrido
        df = soc_hybrid_df(df, Q_Ah=pack.Q_Ah, dt=cfg.sample_dt, alpha=0.85, soc0=100.0, n_series=pack.N_series)
        # 3) Guardar log CSV
        csv_path = cfg.out_dir / f"{name}_log.csv"
        df.to_csv(csv_path, index=False)
        # 4) KPIs
        kpis = compute_kpis(df, pack)
        kpis["Escenario"] = name.capitalize()
        kpis_rows.append(kpis)
        # 5) Gráficas
        plot_soc(df, f"SOC verdadero vs estimado — {name}", cfg.out_dir / f"{name}_soc.png")
        plot_vi(df,  f"Tensión y Corriente — {name}",      cfg.out_dir / f"{name}_vi.png")
        # Potencia/Energía (inline para evitar duplicar función)
        if "P" in df and "E_Wh" in df and _HAVE_PLOT:
            fig, ax1 = plt.subplots(figsize=(7, 3))
            ax1.plot(df["t_s"], df["P"]); ax1.set_xlabel("Tiempo (s)"); ax1.set_ylabel("Potencia (W)")
            ax2 = ax1.twinx(); ax2.plot(df["t_s"], df["E_Wh"], linestyle="--"); ax2.set_ylabel("Energía (Wh)")
            fig.tight_layout(); fig.suptitle(f"Potencia y Energía — {name}", y=1.02, fontsize=10)
            fig.savefig(cfg.out_dir / f"{name}_power_energy.png", dpi=200, bbox_inches="tight")
            plt.close(fig)
        # Error SOC
        plot_soc_error(df, f"Error SOC (estimado - verdadero) — {name}", cfg.out_dir / f"{name}_soc_error.png")

    # Exportar KPIs agregado
    if pd is not None:
        kdf = pd.DataFrame(kpis_rows)[[
            c for c in ["Escenario","E_Wh_net","Loss_Wh","Vmin","Vmax","SOC_final_true", 
            "SOC_final_est","SOC_MAE_%","SOC_RMSE_%"]
            if any(c in r for r in kpis_rows)
        ]]
        kdf.to_csv(cfg.out_dir / "kpis.csv", index=False)
    save_kpis_table_md(kpis_rows, cfg.out_dir / "kpis.md")
    save_json({"kpis": kpis_rows}, cfg.out_dir / "kpis.json")

    # Subida opcional a Firebase
    if cfg.firebase_enable:
        app = firebase_init(cfg)
        if app:
            firebase_upload_doc("bms_runs", cfg.project_tag, {"last_run": time.strftime("%Y-%m-%d %H:%M:%S")})
            for f in ["urbano_log.csv", "extraurbano_log.csv", "kpis.csv", "kpis.json", "kpis.md",
                      "urbano_soc.png", "urbano_vi.png", "urbano_power_energy.png", "urbano_soc_error.png",
                      "extraurbano_soc.png", "extraurbano_vi.png", "extraurbano_power_energy.png", "extraurbano_soc_error.png"]:
                p = cfg.out_dir / f
                if p.exists():
                    firebase_upload_file(p, f"{cfg.project_tag}/{f}")
    print(f"[OK] Simulación completada. Resultados en: {cfg.out_dir.resolve()}")


def run_ingest_csv(cfg: Config) -> None:
    """
    Procesa un CSV existente de banco de pruebas:
      - Estandariza columnas y calcula E_Wh si falta.
      - Estima SOC híbrido con dt promedio.
      - Calcula KPIs y genera gráficas.
    """
    if not cfg.ingest_csv or not cfg.ingest_csv.exists():
        raise FileNotFoundError(f"No se encontró CSV: {cfg.ingest_csv}")
    ensure_out_dir(cfg)
    df = ingest_csv(cfg.ingest_csv)

    dt_est = float(df["t_s"].diff().fillna(0.0).mean() or cfg.sample_dt)
    df2 = soc_hybrid_df(df, Q_Ah=20.0, dt=dt_est, alpha=0.85, soc0=100.0, n_series=16)
    df2.to_csv(cfg.out_dir / f"{cfg.ingest_csv.stem}_soc.csv", index=False)

    pack = LFPTheveninPack()
    k = compute_kpis(df2, pack)
    k["Escenario"] = "BancoCSV"
    save_json({"kpis": k}, cfg.out_dir / f"{cfg.ingest_csv.stem}_kpis.json")
    if _HAVE_TABULATE:
        save_kpis_table_md([k], cfg.out_dir / f"{cfg.ingest_csv.stem}_kpis.md")

    plot_soc(df2, "SOC estimado — Banco CSV", cfg.out_dir / f"{cfg.ingest_csv.stem}_soc.png")
    plot_vi(df2,  "Tensión y Corriente — Banco CSV", cfg.out_dir / f"{cfg.ingest_csv.stem}_vi.png")
    if "P" in df2 and "E_Wh" in df2 and _HAVE_PLOT:
        fig, ax1 = plt.subplots(figsize=(7, 3))
        ax1.plot(df2["t_s"], df2["P"]); ax1.set_xlabel("Tiempo (s)"); ax1.set_ylabel("Potencia (W)")
        ax2 = ax1.twinx(); ax2.plot(df2["t_s"], df2["E_Wh"], linestyle="--"); ax2.set_ylabel("Energía (Wh)")
        fig.tight_layout(); fig.suptitle("Potencia y Energía — Banco CSV", y=1.02, fontsize=10)
        fig.savefig(cfg.out_dir / f"{cfg.ingest_csv.stem}_power_energy.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    print(f"[OK] Ingesta CSV completada. Resultados en: {cfg.out_dir.resolve()}")


def run_ingest_serial(cfg: Config, duration_s: Optional[int]) -> None:
    """
    Captura por puerto serie y luego procesa el CSV resultante con 'run_ingest_csv'.
    """
    ensure_out_dir(cfg)
    out_csv = cfg.out_dir / "serial_log.csv"
    ingest_serial_to_csv(cfg, out_csv, duration_s=duration_s)
    cfg2 = Config(**{**cfg.__dict__, "ingest_csv": out_csv})
    run_ingest_csv(cfg2)


# --------------------------- 13. INTERFAZ DE LÍNEA (CLI) ---------------------
def build_argparser() -> argparse.ArgumentParser:
    """
    Define la CLI del programa con 3 modos:
      - simulate     : corre suite de simulación
      - ingest-csv   : procesa un CSV existente
      - ingest-serial: captura por serie y procesa
    Incluye parámetros de salida, dt y opciones de Firebase.
    """
    p = argparse.ArgumentParser(
        description="BMS (IMPI) – simulación/ingesta/estimación/KPIs/gráficas."
    )
    p.add_argument("--mode", choices=["simulate", "ingest-csv", "ingest-serial"], default="simulate",
                   help="Modo de ejecución.")
    p.add_argument("--out", type=str, default="out", help="Directorio de salida.")
    p.add_argument("--csv", type=str, help="Ruta a CSV para --mode ingest-csv.")
    p.add_argument("--serial-port", type=str, default="/dev/ttyACM0", help="Puerto serie (ingest-serial).")
    p.add_argument("--baud", type=int, default=115200, help="Baudios (ingest-serial).")
    p.add_argument("--duration", type=int, default=None, help="Duración de captura serie en segundos.")
    p.add_argument("--dt", type=float, default=1.0, help="Paso dt (s) para simulación/estimación.")
    # Firebase
    p.add_argument("--firebase", action="store_true", help="Habilita subida a Firebase.")
    p.add_argument("--fb-creds", type=str, help="Credenciales JSON de Firebase.")
    p.add_argument("--fb-bucket", type=str, help="Nombre del bucket de Firebase Storage.")
    p.add_argument("--tag", type=str, default="BMS_LFP_48V20Ah", help="Etiqueta de proyecto/remoto.")
    return p


def parse_cfg(args: argparse.Namespace) -> Config:
    """
    Convierte argumentos CLI en un objeto Config.
    """
    return Config(
        out_dir=Path(args.out),
        simulate=(args.mode == "simulate"),
        ingest_csv=Path(args.csv) if args.csv else None,
        ingest_serial=(args.mode == "ingest-serial"),
        serial_port=args.serial_port,
        serial_baud=args.baud,
        sample_dt=args.dt,
        firebase_enable=bool(args.firebase),
        firebase_creds_json=Path(args.fb_creds) if args.fb_creds else None,
        firebase_bucket=args.fb_bucket,
        project_tag=args.tag
    )
def main():
    """
    Punto de entrada principal. Selecciona el flujo según --mode.
    """
    ap = build_argparser()
    args = ap.parse_args()
    cfg = parse_cfg(args)

    if args.mode == "simulate":
        run_simulation_suite(cfg)
    elif args.mode == "ingest-csv":
        if not cfg.ingest_csv:
            raise SystemExit("Debes pasar --csv=archivo.csv para ingest-csv.")
        run_ingest_csv(cfg)
    elif args.mode == "ingest-serial":
        run_ingest_serial(cfg, duration_s=args.duration)
    else:
        raise SystemExit("Modo no reconocido.")
if __name__ == "__main__":
    main()