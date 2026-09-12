"""Laboratorio 5. Módulo de funciones para interactuar con el Arcade Learning Environment (ALE).

Funciones:
    crear_entorno          crea un entorno de Gymnasium y, si se indica video_folder, lo envuelve con RecordVideo
    agente_aleatorio       política de referencia: acción muestreada de env.action_space
    agente_regla_simple    política heurística para Space Invaders basada en la imagen
    ejecutar_episodio      corre un episodio completo y devuelve pasos y recompensa total
    generar_video_agente   crea el entorno con grabación, corre n episodios, cierra el entorno y devuelve rutas y métricas
"""
from __future__ import annotations

import os
from typing import Callable

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo

import ale_py

gym.register_envs(ale_py)


def crear_entorno(
    nombre_entorno: str = "ALE/SpaceInvaders-v5",
    video_folder: str | None = None,
    episode_trigger: Callable[[int], bool] | None = None,
    name_prefix: str = "video",
    render_mode: str | None = None,
    fps: int | None = None,
    **kwargs,
) -> gym.Env:
    """Crea y retorna un entorno de Gymnasium.

    Si se especifica video_folder, el entorno se envuelve con gymnasium.wrappers.RecordVideo
    para grabar los episodios que indique episode_trigger (por defecto, todos).
    kwargs se pasan a gym.make, por ejemplo obs_type, frameskip, repeat_action_probability
    o full_action_space en los entornos de ALE.
    """
    if video_folder is not None:
        render_mode = "rgb_array"
    env = gym.make(nombre_entorno, render_mode=render_mode, **kwargs)
    if video_folder is not None:
        if episode_trigger is None:
            episode_trigger = lambda episodio: True
        env = RecordVideo(
            env,
            video_folder=video_folder,
            episode_trigger=episode_trigger,
            name_prefix=name_prefix,
            fps=fps,
            disable_logger=True,
        )
    return env


def agente_aleatorio(observation, env: gym.Env) -> int:
    """Agente de referencia: acción uniforme al azar del espacio de acciones."""
    return env.action_space.sample()


COLOR_JUGADOR = np.array([50, 132, 50], dtype=np.uint8)
COLOR_ALIEN = np.array([134, 134, 29], dtype=np.uint8)
ACCION_FIRE, ACCION_RIGHTFIRE, ACCION_LEFTFIRE = 1, 4, 5


def agente_regla_simple(observation, env: gym.Env, margen: int = 2) -> int:
    """Regla fija para ALE/SpaceInvaders-v5 con observación RGB.

    Localiza el cañón del jugador por su color en las filas 185 a 195 y los invasores por su
    color en las filas 20 a 150. Calcula el centro horizontal de la fila más baja de invasores,
    se desplaza hacia él disparando y, cuando está alineado, solo dispara.
    Si la observación no es una imagen RGB de Atari, devuelve una acción aleatoria.
    """
    obs = np.asarray(observation)
    if obs.ndim != 3 or obs.shape[0] < 195 or obs.shape[2] != 3:
        return env.action_space.sample()
    xs_jugador = np.nonzero(np.all(obs[185:195] == COLOR_JUGADOR, axis=-1))[1]
    ys_alien, xs_alien = np.nonzero(np.all(obs[20:150] == COLOR_ALIEN, axis=-1))
    if len(xs_jugador) == 0 or len(xs_alien) == 0:
        return ACCION_FIRE
    x_jugador = xs_jugador.mean()
    fila_baja = ys_alien >= ys_alien.max() - 8
    objetivo = xs_alien[fila_baja].mean()
    if x_jugador < objetivo - margen:
        return ACCION_RIGHTFIRE
    if x_jugador > objetivo + margen:
        return ACCION_LEFTFIRE
    return ACCION_FIRE


def ejecutar_episodio(env: gym.Env, funcion_agente: Callable, max_steps: int = 10000, seed: int | None = None) -> dict:
    """Ejecuta un episodio completo hasta terminated, truncated o max_steps.

    Retorna un diccionario con pasos, recompensa total, terminated y truncated.
    """
    observation, info = env.reset(seed=seed)
    if seed is not None:
        env.action_space.seed(seed)  # hace reproducible también al agente aleatorio
    pasos, recompensa = 0, 0.0
    terminated = truncated = False
    while not (terminated or truncated) and pasos < max_steps:
        accion = funcion_agente(observation, env)
        observation, reward, terminated, truncated, info = env.step(accion)
        recompensa += float(reward)
        pasos += 1
    return {"pasos": pasos, "recompensa": recompensa, "terminated": bool(terminated), "truncated": bool(truncated)}


def generar_video_agente(
    nombre_entorno: str,
    funcion_agente: Callable,
    video_folder: str,
    name_prefix: str,
    n_episodios: int = 1,
    max_steps: int = 10000,
    seed: int | None = None,
    **kwargs,
) -> tuple[list[str], list[dict]]:
    """Crea el entorno con grabación de video, ejecuta n_episodios y cierra el entorno.

    Retorna la lista de rutas de los videos generados y la lista de métricas de cada episodio.
    env.close() es indispensable: RecordVideo escribe el archivo al cerrar el episodio o el entorno.
    """
    env = crear_entorno(nombre_entorno, video_folder=video_folder, episode_trigger=lambda e: True,
                        name_prefix=name_prefix, **kwargs)
    metricas = []
    try:
        for i in range(n_episodios):
            m = ejecutar_episodio(env, funcion_agente, max_steps=max_steps, seed=None if seed is None else seed + i)
            m["episodio"] = i
            metricas.append(m)
    finally:
        env.close()
    rutas = [os.path.join(video_folder, f"{name_prefix}-episode-{i}.mp4") for i in range(n_episodios)]
    rutas = [r for r in rutas if os.path.exists(r)]
    return rutas, metricas
