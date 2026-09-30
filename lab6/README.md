# Laboratorio 6. Transfer Learning y Fine-Tuning sobre CIFAR-10

CC3092 Deep Learning y Sistemas Inteligentes. Brandon Werner Rivera Cabrera 23088.

Comparación de tres estrategias de clasificación en CIFAR-10:

| Modelo | Entrada | Qué se entrena | Iteraciones |
|---|---|---|---|
| CNN desde cero | 32×32 | toda la red | CNN-1, CNN-2, CNN-3 |
| VGG-16 feature extractor | 128×128 | solo el clasificador | FE-1, FE-2, FE-3 |
| VGG-16 fine-tuning | 128×128 | bloques superiores + clasificador | FT-1, FT-2, FT-3, FT-4 |

Todo está en `Lab6_Transfer_Learning.ipynb`: exploración del dataset, pipelines de preprocesamiento,
investigación de las herramientas de PyTorch, las 10 iteraciones, la evaluación única en test con
matrices de confusión, el experimento con el 10 % de los datos y la medición de recursos (parámetros,
MACs/FLOPs, latencia, memoria pico y tiempo).

## Ejecución (Windows, PowerShell)

```powershell
cd $HOME\Documents\Laboratorios-Deep-Learning\lab6
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 Lab6_Transfer_Learning.ipynb
```

PyTorch con CUDA debe instalarse antes desde https://download.pytorch.org/whl/cu130 (ver la sección 2.1
del README del Proyecto 2). El notebook descarga CIFAR-10 (170 MB) en `data/` y los pesos de VGG-16
(528 MB) en la caché de PyTorch la primera vez.

## Salidas

- `results/iteraciones.csv`: una fila por iteración (configuración, parámetros, mejor epoch, métricas de validación, tiempo, memoria).
- `results/historial.csv`: pérdida y métricas por epoch de todas las iteraciones.
- `results/test.json`: métricas de test, matrices de confusión y pares de clases más confundidos.
- `results/datos_10pct.csv`: experimento con el 10 % de los datos.
- `results/comparacion.csv`: tabla de rendimiento y recursos de los tres modelos finales.
- `results/costo_por_resolucion.csv` y `results/entorno.json`: costo de VGG-16 por resolución y hardware usado.
- `figs/`: todas las figuras.

`data/` y `checkpoints/` no se suben al repositorio.

## Decisiones de implementación

El preprocesamiento y la augmentation se hacen en la GPU sobre tensores `uint8` (recorte aleatorio con
relleno de 4 píxeles y volteo horizontal como una transformación afín por lote, con el redimensionamiento
a 128×128 incluido para VGG-16). Esto evita los procesos de `DataLoader`, que dentro de Jupyter en
Windows suelen fallar, y hace que redimensionar no sea un cuello de botella. El entrenamiento usa
precisión mixta bfloat16 y `channels_last`.
