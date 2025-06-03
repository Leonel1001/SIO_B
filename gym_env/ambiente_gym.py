import gymnasium as gym
from gymnasium import spaces
import numpy as np
from ambiente import Ambiente  # o teu ambiente original

class CustomAmbiente(gym.Env):
    def __init__(self, mapa, destino):
        super(CustomAmbiente, self).__init__()

        self.mapa = mapa
        self.destino = destino
        self.env = Ambiente(mapa, destino)
        self.visitados = set()
        self.passos_repetidos = 0

        # Espaço de observação normalizado entre 0 e 1
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(14,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)

    def normalize_estado(self, estado):
        estado = np.array(estado, dtype=np.float32)
        divisores = np.array([
            1.0, 1.0,    # dist_y, dist_x
            1.0, 1.0,    # pos_y, pos_x
            1.0, 1.0, 1.0, 1.0,  # obstáculos
            3.0, 3.0,    # tam_y, tam_x
            1.0, 1.0,    # euclidiana, manhattan
            1.0, 1.0     # direção y, x
        ], dtype=np.float32)
        return np.clip(estado / divisores, 0.0, 1.0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        estado = self.env.reset()
        self.env.ultima_distancia = np.linalg.norm(np.array(self.env.estado) - np.array(self.env.objetivo))
        self.visitados = set()
        self.visitados.add(tuple(self.env.estado))
        self.passos_repetidos = 0
        return self.normalize_estado(estado), {}

    def step(self, action):
        estado_anterior = tuple(self.env.estado)
        prox_estado, _, done, info = self.env.step(action)
        atual = tuple(self.env.estado)

        dist_atual = np.linalg.norm(np.array(self.env.estado) - np.array(self.env.objetivo))
        delta_dist = self.env.ultima_distancia - dist_atual
        recompensa = -0.1 + (delta_dist * 2)
        self.env.ultima_distancia = dist_atual

        if atual in self.visitados:
            self.passos_repetidos += 1
            recompensa -= 2  # penaliza repetir
        else:
            self.visitados.add(atual)
            self.passos_repetidos = 0

        if self.passos_repetidos >= 10 and self.env.estado != self.env.objetivo:
            recompensa -= 10
            done = True


        if self.env.estado == self.env.objetivo:
            recompensa = 200
            done = True

        return self.normalize_estado(prox_estado), recompensa, done, False, {"distancia": dist_atual}

    def render(self):
        pass