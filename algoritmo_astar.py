import heapq

def heuristica(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan distance

def vizinhos(pos, mapa):
    direcoes = [(-1,0), (1,0), (0,-1), (0,1)]  # cima, baixo, esquerda, direita
    resultado = []
    for dx, dy in direcoes:
        nx, ny = pos[0] + dx, pos[1] + dy
        if 0 <= nx < len(mapa) and 0 <= ny < len(mapa[0]) and mapa[nx][ny] == 0:
            resultado.append((nx, ny))
    return resultado

def a_star(mapa, inicio, objetivo):
    fila = []
    heapq.heappush(fila, (0, inicio))
    veio_de = {inicio: None}
    custo_ate_agora = {inicio: 0}

    while fila:
        _, atual = heapq.heappop(fila)

        if atual == objetivo:
            break

        for prox in vizinhos(atual, mapa):
            novo_custo = custo_ate_agora[atual] + 1
            if prox not in custo_ate_agora or novo_custo < custo_ate_agora[prox]:
                custo_ate_agora[prox] = novo_custo
                prioridade = novo_custo + heuristica(prox, objetivo)
                heapq.heappush(fila, (prioridade, prox))
                veio_de[prox] = atual

    # Reconstruir o caminho
    caminho = []
    atual = objetivo
    while atual != inicio:
        caminho.append(atual)
        atual = veio_de.get(atual)
        if atual is None:
            return []  # sem caminho
    caminho.append(inicio)
    caminho.reverse()
    return caminho

# 👇 Isto só será executado se correres ESTE ficheiro diretamente
if __name__ == "__main__":
    from mapa_hard import criar_mapa_hard
    import numpy as np
    import matplotlib.pyplot as plt

    def mostrar_caminho(mapa, caminho):
        matriz = np.array(mapa)
        for x, y in caminho:
            matriz[x][y] = 2
        plt.imshow(matriz, cmap='gray_r')
        plt.title("Caminho A* no mapa_hard")
        plt.show()

    mapa = criar_mapa_hard()
    inicio = (0, 0)
    objetivo = (29, 29)

    if mapa[inicio[0]][inicio[1]] == 1 or mapa[objetivo[0]][objetivo[1]] == 1:
        raise ValueError("Início ou objetivo está num obstáculo!")

    caminho = a_star(mapa, inicio, objetivo)
    print(f"Caminho encontrado com {len(caminho)} passos:")
    print(caminho)
    mostrar_caminho(mapa, caminho)
