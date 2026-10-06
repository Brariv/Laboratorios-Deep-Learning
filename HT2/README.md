# Hoja de trabajo 2. Transformers y mecanismos de atención

CC3092 Deep Learning y Sistemas Inteligentes. Brandon Werner Rivera Cabrera 23088.

`HT2_Transformers_Atencion.ipynb` contiene:

1. Scaled dot-product attention y multi-head attention implementadas desde cero, un ejercicio a mano paso a paso y verificaciones contra `F.scaled_dot_product_attention` y `nn.MultiheadAttention`, varianza del producto punto, saturación de la softmax, codificación posicional sinusoidal, equivarianza a permutaciones y Post-LN frente a Pre-LN.
2. Escalamiento del tiempo, la memoria y los FLOPs de la atención con la longitud de la secuencia, comparando la implementación propia, `F.scaled_dot_product_attention` y atención lineal.
3. Atención en `bert-base-uncased` y `gpt2`: mapas de calor de cabezas con patrones identificados, correferencia de «it», máscara causal y attention sinks, entropía por capa, ablación de cabezas de GPT-2 y comparación de la atención con la saliencia por gradiente.

## Ejecución (macOS)

```bash
cd ~/Documents/UVG/ia
source .venv/bin/activate
pip install -r HT2/requirements.txt
cd HT2
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 HT2_Transformers_Atencion.ipynb
```

La primera ejecución descarga BERT y GPT-2 desde Hugging Face. Los benchmarks usan la GPU de la Mac (MPS) si está disponible.

## Salidas

- `results/`: ejercicio a mano, verificaciones, escalamiento, patrones por cabeza, correferencia, attention sinks, entropía por capa, ablación de cabezas y atención frente a saliencia.
- `figs/`: todas las figuras.
