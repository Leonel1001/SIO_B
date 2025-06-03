import gymnasium as gym
from gymnasium import spaces
import numpy as np
from ambiente import Ambiente
from mapa_hard import criar_mapa_hard

class CustomAmbienteHardDinamico(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(14,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self.env = None  # será criado no reset
        self.visitados = set()
        self.passos_repetidos = 0

    def _novo_mapa_e_objetivo(self):
        mapa = criar_mapa_hard()
        while True:
            destino = (np.random.randint(0, len(mapa)), np.random.randint(0, len(mapa[0])))
            if mapa[destino[0]][destino[1]] == 0:
                return mapa, destino

    def normalize_estado(self, estado):
        estado = np.array(estado, dtype=np.float32)
        divisores = np.array([
            1.0, 1.0,  # dist_y, dist_x
            1.0, 1.0,  # pos_y, pos_x
            1.0, 1.0, 1.0, 1.0,
            3.0, 3.0,  # tam_y, tam_x
            1.0, 1.0,  # euclidiana, manhattan
            1.0, 1.0   # dir_y, dir_x
        ], dtype=np.float32)
        return np.clip(estado / divisores, 0.0, 1.0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mapa, destino = self._novo_mapa_e_objetivo()
        self.env = Ambiente(mapa, destino)
        estado = self.env.reset()
        self.env.ultima_distancia = np.linalg.norm(np.array(self.env.estado) - np.array(self.env.objetivo))
        self.visitados = set([tuple(self.env.estado)])
        self.passos_repetidos = 0
        return self.normalize_estado(estado), {}

    def step(self, action):
        atual = tuple(self.env.estado)
        prox_estado, _, done, _ = self.env.step(action)
        novo = tuple(self.env.estado)

        dist_atual = np.linalg.norm(np.array(self.env.estado) - np.array(self.env.objetivo))
        delta_dist = self.env.ultima_distancia - dist_atual
        recompensa = -0.1 + (delta_dist * 2)
        self.env.ultima_distancia = dist_atual

        if novo in self.visitados:
            self.passos_repetidos += 1
            recompensa -= 2
        else:
            self.visitados.add(novo)
            self.passos_repetidos = 0

        if self.passos_repetidos >= 10:
            recompensa -= 10
            done = True

        if self.env.estado == self.env.objetivo:
            recompensa = 200
            done = True

        return self.normalize_estado(prox_estado), recompensa, done, False, {"distancia": dist_atual}

    def render(self):
        pass
