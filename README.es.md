[[English](README.md) ] | [Español ]

# Línea de producción Bosch: ¿qué piezas van a fallar el control de calidad?

![CI](https://github.com/Rxyxs/bosch-line-quality/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10-blue)
![LightGBM](https://img.shields.io/badge/LightGBM-4-green)
![License](https://img.shields.io/badge/license-MIT-green)

1,18 millones de piezas reales de una línea de ensamblaje de Bosch, cada una con hasta 4.264
mediciones anónimas y marcas de tiempo en 52 estaciones, y un 0,58% que falla el control de
calidad final. Datos de la competencia de Kaggle
[Bosch Production Line Performance](https://www.kaggle.com/competitions/bosch-production-line-performance).

La pregunta que me interesa es la que se haría un ingeniero de calidad: **si solo podemos
inspeccionar un pequeño porcentaje de las piezas, ¿cuáles, y cuánto se ahorra?**

**Late submission en Kaggle:** MCC **0,400** en el público y **0,418** en el privado (métrica de
la competencia; la solución ganadora obtuvo cerca de 0,52).

## Resultados

LightGBM con tres conjuntos de features, cada uno medido de dos formas: CV estratificado de 5
folds sobre todas las piezas, y un holdout temporal (entreno con el 80% más antiguo por fecha de
inicio y mido sobre el 20% más reciente).

| Features | MCC en CV | MCC en holdout temporal |
|---|---|---|
| 968 mediciones | 0,228 | 0,061 |
| + proceso (ruta, estaciones, tiempo por línea) | 0,231 | 0,038 |
| + secuencia (posición en el flujo de producción) | **0,402** | **0,207** |

![CV frente a holdout temporal](docs/img/validation.png)

**1. Las fallas vienen por lotes, no por pieza.** Entre las piezas producidas junto a una pieza
fallada, la tasa de falla es 6,4%, 11 veces la tasa base. Las dos features más importantes, por
lejos, son `id_gap_next` y `same_lot_next`: si la pieza siguiente tiene un Id consecutivo y la
misma hora de inicio, o sea, en qué lugar del lote está la pieza. Las mediciones pesan mucho
menos; solo unas pocas de las estaciones S29, S24 y S33 aparecen entre las principales.

**2. La validación cruzada exagera lo que el modelo logra con piezas futuras.** El CV mezcla
piezas de todo el período, así que el modelo aprende patrones propios de cada época y se evalúa
sobre esas mismas épocas. Sobre piezas producidas después, el modelo de solo mediciones cae de
0,228 a 0,061. Las features de secuencia son las únicas que se sostienen (0,207), y su umbral,
elegido en el período anterior, sigue siendo casi óptimo en el posterior (0,207 frente a 0,208
con el mejor umbral posible); con los otros conjuntos se pierde tanto el orden como el umbral.

**3. Las features de proceso no aportan por sí solas.** Ruta, número de estaciones y tiempos de
permanencia apenas mueven el CV (+0,003) y empeoran el resultado temporal.

El test de Kaggle está intercalado en el tiempo con las piezas de entrenamiento, por eso el
leaderboard (0,418) se parece al CV (0,402) y no al holdout temporal. Un modelo puesto en la
línea vería el número temporal.

## Del MCC a una política de inspección

Ordeno las piezas por riesgo predicho y saco las más riesgosas para inspección manual:

| Piezas inspeccionadas | Fallas atrapadas (CV) | Fallas atrapadas (piezas posteriores) |
|---|---|---|
| 1% | 39% | 22% |
| 2% | 52% | 32% |
| 5% | 74% | 61% |
| 10% | 80% | 77% |

![Curva de captura](docs/img/capture.png)

Con un costo de 1 por inspección y de R por cada falla que escapa, la política que minimiza el
costo total inspecciona las k piezas más riesgosas. Si una falla que escapa cuesta 100 veces una
inspección, la mejor política sobre piezas posteriores inspecciona el 6,6% de la producción y
baja el costo de calidad al **40%** de no inspeccionar nada (33% según el CV). La tabla completa
para R entre 10 y 1.000 está en [`reports/inspection.json`](reports/inspection.json).

## Método

```
CSV de train/test (14 GB) ──► Parquet con esquema explícito (2,3 GB, 5 min)
        │
        ├── numeric: las 968 mediciones tal cual
        ├── process: estaciones visitadas, tiempo por línea, número de mediciones, frecuencia de la ruta
        └── sequence: orden por (hora de inicio, Id) sobre train+test; salto de Id y de tiempo a
                      las vecinas, tamaño y posición del lote, indicadores de mismo lote
                      (solo tiempos e Ids; nunca la etiqueta de una vecina, así que no hay fuga)
        │
LightGBM (early stopping en un 10% interno, nunca en el fold que se evalúa)
        │
umbral que maximiza el MCC fuera de la muestra ──► submission.csv (promedio de 3 semillas)
        │
curva de captura y política de inspección de costo mínimo
```

## Datos

[Bosch Production Line Performance](https://www.kaggle.com/competitions/bosch-production-line-performance),
Kaggle, 2016. Descargados en octubre de 2026 bajo las reglas de la competencia, que no permiten
redistribuirlos; los datos no están en este repositorio.

## Reproducir

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
python -m bosch.ingest      # requiere ~/.kaggle/kaggle.json y aceptar las reglas
python -m bosch.features
python -m bosch.model       # tres conjuntos de features, CV + holdout temporal (~15 min)
python -m bosch.submit      # modelo final y submission.csv
python -m bosch.analysis    # política de inspección y figuras
pytest -q tests
```

## Limitaciones

- Las features son anónimas, así que los hallazgos sobre mediciones no se pueden atribuir a
  magnitudes físicas.
- Una sola familia de modelos y un solo set de hiperparámetros; sin ajuste más allá del early
  stopping.
- El modelo de costos supone que la inspección detecta toda falla que revisa, y un costo fijo
  por falla que escapa.

## Licencia

Código: MIT. Datos: reglas de la competencia de Kaggle (no se redistribuyen).
