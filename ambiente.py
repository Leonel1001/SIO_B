import numpy as np


class Ambiente:
    def __init__(self, mapa, objetivo=None):
        self.mapa = np.array(mapa)
        self.inicio = (0, 0)
        self.objetivo = objetivo if objetivo is not None else (5, 5)
        self.estado = self.inicio
        self.pos_agente = self.inicio  # Inicializa a posição do agente

    def reset(self):
        self.estado = self.inicio
        self.pos_agente = self.inicio

        # Calcula a distância máxima entre início e objetivo
        self.distancia_maxima = np.linalg.norm(np.array(self.pos_agente) - np.array(self.objetivo))

        return self._obter_estado()


    def step(self, acao):
        y, x = self.estado
        movimentos = [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]
        novo_estado = movimentos[acao]

        # Verificar se o novo estado está dentro dos limites do mapa
        if (
            0 <= novo_estado[0] < self.mapa.shape[0]
            and 0 <= novo_estado[1] < self.mapa.shape[1]
        ):
            if self.mapa[novo_estado] != 1:  # Verificar se não é um obstáculo
                self.estado = novo_estado
                self.pos_agente = novo_estado  # Atualiza a posição do agente

        recompensa = -1
        done = False
        if self.estado == self.objetivo:
            recompensa = 200  # Recompensa maior por alcançar o objetivo
            done = True

        return self._obter_estado(), recompensa, done, {}

    def _obter_estado(self):
        y, x = self.estado
        gy, gx = self.objetivo
        altura, largura = self.mapa.shape

        # Diferença de posição
        dist_y = (gy - y) / altura
        dist_x = (gx - x) / largura

        # Posição atual normalizada
        pos_y = y / altura
        pos_x = x / largura

        # Obstáculos nas 4 direções
        obstaculos = [
            self._obstaculo(y - 1, x),  # Cima
            self._obstaculo(y + 1, x),  # Baixo
            self._obstaculo(y, x - 1),  # Esquerda
            self._obstaculo(y, x + 1),  # Direita
        ]

        # Tamanho do mapa (normalizado)
        tam_y = altura / 10
        tam_x = largura / 10

        # Distância euclidiana normalizada
        euclidiana = np.linalg.norm([gy - y, gx - x]) / np.linalg.norm([altura, largura])

        # Distância de Manhattan normalizada
        manhattan = (abs(gy - y) + abs(gx - x)) / (altura + largura)

        # Direção do objetivo (vetor normalizado)
        delta = np.array([gy - y, gx - x], dtype=np.float32)
        norm = np.linalg.norm(delta)
        if norm > 0:
            dir_y, dir_x = delta / norm
        else:
            dir_y, dir_x = 0.0, 0.0

        return np.array([
            dist_y, dist_x,
            pos_y, pos_x,
            *obstaculos,
            tam_y, tam_x,
            euclidiana,
            manhattan,
            dir_y, dir_x
        ], dtype=np.float32)



    def _obstaculo(self, y, x):
        if 0 <= y < self.mapa.shape[0] and 0 <= x < self.mapa.shape[1]:
            return 1 if self.mapa[y][x] == 1 else 0
        return 1  # Considera fora do mapa como obstáculo

