# Laboratorio 7. NLP end-to-end y Embeddings

CC3092 Deep Learning y Sistemas Inteligentes. Brandon Werner Rivera Cabrera 23088.

Pipeline completo texto → tokens → IDs → vectores → modelo → salida:

1. Exploración y preprocesamiento de WikiText-103: normalización compatible con GloVe, tokenizador propio comparado con NLTK, ley de Zipf, vocabulario por `min_count`, submuestreo y pares de skip-gram.
2. Skip-gram con negative sampling implementado en PyTorch (dos tablas `nn.Embedding` dispersas y `SparseAdam`), entrenado sobre un subconjunto de 25 M tokens con 12 iteraciones más una configuración final.
3. Word2Vec de gensim con el mismo corpus, vocabulario e hiperparámetros, y GloVe `glove-wiki-gigaword-100` como referencia.
4. Aritmética vectorial: analogías individuales con 3CosAdd propio verificado contra `most_similar`, paralelismo de vectores diferencia, las 14 categorías de `questions-words.txt` con 3CosAdd y 3CosMul, y t-SNE.
5. Clasificación de AG News: TF-IDF + regresión logística, `EmbeddingBag` aleatorio y con cada conjunto de embeddings congelado o con fine-tuning, evaluación única en test y curva de F1 contra fracción de datos.

Todo está en `Lab7_NLP_Embeddings.ipynb`.

## Ejecución (Windows, PowerShell)

```powershell
cd $HOME\Documents\Laboratorios-Deep-Learning\lab7
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 Lab7_NLP_Embeddings.ipynb
```

PyTorch con CUDA debe estar instalado antes. La primera ejecución descarga WikiText-103 y AG News desde Hugging Face y GloVe (128 MB) con `gensim.downloader`.

## Salidas

- `results/`: estadísticas del corpus, Zipf, vocabulario por umbral, submuestreo, pares, iteraciones y curvas de SGNS, vecinos por época, analogías individuales y por categoría, paralelismo, clasificación en validación y test, fracciones de datos y tabla comparativa.
- `figs/`: todas las figuras.
- `vectores/sgns_final.kv`: vectores del mejor modelo SGNS en formato `KeyedVectors` de gensim. Se cargan con `KeyedVectors.load("vectores/sgns_final.kv")`.
- `datos/`: corpus preprocesado, no se sube al repositorio.
