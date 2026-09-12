# Laboratorio 5. Agentes en el Arcade Learning Environment (ALE): Space Invaders

CC3092 Deep Learning y Sistemas Inteligentes. Brandon Werner Rivera Cabrera 23088.

Módulo de funciones reutilizables para crear entornos de Gymnasium, ejecutar agentes y grabar video
de las partidas. Es la base del Proyecto 2, donde la función de agente aleatorio se reemplaza por la
política aprendida.

## Contenido

```
lab5/
├── Lab5_ALE_Space_Invaders.ipynb   notebook con la investigación, el módulo y los resultados (ejecutado)
├── ale_utils.py                    módulo importable con las cinco funciones
├── Lab 5 Deep Learning.docx        informe escrito
├── Lab 5 Deep Learning.pdf         el mismo informe en PDF (entregable)
├── videos/                         mp4 generados con RecordVideo
└── figs/                           figuras usadas en el informe
```

## Uso

```bash
pip install "gymnasium>=1.1" "ale-py>=0.11" opencv-python imageio imageio-ffmpeg matplotlib pandas
jupyter lab Lab5_ALE_Space_Invaders.ipynb
```

O directamente desde el módulo:

```python
from ale_utils import (crear_entorno, agente_aleatorio, agente_regla_simple,
                       ejecutar_episodio, generar_video_agente)

env = crear_entorno("ALE/SpaceInvaders-v5")
print(ejecutar_episodio(env, agente_regla_simple, seed=0))
env.close()

rutas, metricas = generar_video_agente("ALE/SpaceInvaders-v5", agente_aleatorio,
                                       video_folder="videos", name_prefix="aleatorio",
                                       n_episodios=3, seed=0)
```

## Funciones

| Función | Qué hace |
|---|---|
| `crear_entorno(nombre_entorno, video_folder=None, episode_trigger=None, name_prefix, **kwargs)` | Crea el entorno con `gym.make` y, si se indica `video_folder`, lo envuelve con `RecordVideo` (fija `render_mode="rgb_array"`). Funciona con cualquier entorno de Gymnasium. |
| `agente_aleatorio(observation, env)` | Acción de `env.action_space.sample()`. Línea base. |
| `agente_regla_simple(observation, env, margen=2)` | Regla sobre la imagen RGB: ubica el cañón y los invasores por su color, apunta al centro de la fila más baja y dispara mientras se alinea. |
| `ejecutar_episodio(env, funcion_agente, max_steps=10000, seed=None)` | Corre un episodio hasta `terminated`, `truncated` o `max_steps`. Devuelve pasos y recompensa total. |
| `generar_video_agente(nombre_entorno, funcion_agente, video_folder, name_prefix, n_episodios=1, seed=None, **kwargs)` | Crea el entorno con grabación, corre los episodios, cierra el entorno (necesario para que el mp4 se escriba) y devuelve rutas y métricas. |

## Resultados

Con semillas fijas, 3 episodios grabados por agente y 10 episodios adicionales para la comparación:

| Agente | Recompensa promedio | Pasos promedio |
|---|---|---|
| Aleatorio | 115.0 | 445.8 |
| Regla simple | 233.0 | 1121.0 |
