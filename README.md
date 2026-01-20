BMS – Sistema de Caracterización y Estimación del Estado de Carga (SOC) para Baterías Li-ion (48V–20Ah)

Herramienta de simulación y análisis de un Sistema de Administración de Energía (BMS) basado en baterías Li-ion (48V–20Ah), utilizando un modelo equivalente de Thévenin, un estimador híbrido de SOC y procesamiento de datos experimentales.

Este repositorio contiene el código fuente oficial destinado al registro ante el Instituto Mexicano de la Propiedad Industrial (IMPI) como: “Caracterización de un sistema de administración de energía (BMS) basado en baterías Li-ion para maximizar la autonomía de vehículos eléctricos”.

Incluye:
	•	Simulación eléctrica completa de un pack LiFePO4 (48V–20Ah).
	•	Estimación híbrida del SOC basada en Coulomb, OCV y filtro complementario.
	•	Procesamiento de datos reales desde archivos CSV o puerto serie.
	•	Cálculo de indicadores clave de desempeño (KPIs).
	•	Exportación automática en CSV, JSON, Markdown y PNG.
	•	Integración opcional con Firebase.

FUNCIONALIDADES PRINCIPALES
	1.	Simulación del pack LiFePO4 (48V–20Ah)

	•	Modelo Thévenin R0–R1–C1.
	•	Escenarios urbano y extraurbano.
	•	Generación automática de V, I, P, energía y SOC.

	2.	Estimación híbrida del SOC

	•	Conteo de Coulomb.
	•	Estimación por OCV inversa.
	•	Filtro complementario para fusionar ambas estimaciones.
	•	Obtención de SOC_est y comparación con SOC_true.

	3.	Ingesta de datos reales

	•	Lectura de CSV con t_ms/t_s, V, I y P.
	•	Estimación automática de energía y normalización de datos.

	4.	Captura desde puerto serie
Formato requerido:
t_ms,V,I,P
	5.	Cálculo de KPIs
Incluye energía neta, pérdidas resistivas, Vmin/Vmax, error MAE y RMSE del SOC.
	6.	Generación de archivos de salida
CSV, JSON, Markdown y PNG (SOC, VI, potencia/energía, error).
	7.	Conexión opcional con Firebase
Subida de archivos y registro de metadatos.

EJECUCIÓN DEL SOFTWARE

Simulación completa:
python3 BMS_LiIon_Caracterizacion_Autonomia_VEHICULOS_ELECTRICOS.py –mode simulate

Procesar un CSV real:
python3 BMS_LiIon_Caracterizacion_Autonomia_VEHICULOS_ELECTRICOS.py –mode ingest-csv –csv archivo.csv

Captura desde puerto serie (120 s):
python3 BMS_LiIon_Caracterizacion_Autonomia_VEHICULOS_ELECTRICOS.py –mode ingest-serial –duration 120

Subida a Firebase:
python3 BMS_LiIon_Caracterizacion_Autonomia_VEHICULOS_ELECTRICOS.py –mode simulate –firebase –fb-creds cred.json –fb-bucket mi-bucket

ESTRUCTURA DEL REPOSITORIO

BMS_IMPI/
	•	README.md
	•	.gitignore
	•	BMS_LiIon_Caracterizacion_Autonomia_VEHICULOS_ELECTRICOS.py
	•	out/ (se genera automáticamente)

AUTORES
	•	M.C. Héctor Javier Jarquín-Flores
	•	M.C. Álvaro César Guevara-Ramírez
	•	Dr. Carlos Mauricio Lastre-Domínguez
	•	Dr. Eric Mario Silva-Cruz

TITULAR DE LOS DERECHOS PATRIMONIALES
Tecnológico Nacional de México / Instituto Tecnológico de Oaxaca

VERSIÓN
Versión 1.1 — Documentación ampliada para registro ante IMPI

LICENCIA
Reservados todos los derechos por los autores. Uso académico y de investigación.
