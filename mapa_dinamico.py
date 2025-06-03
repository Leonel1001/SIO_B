import random


def criar_mapa_dinamico():
    largura, altura = 30, 30
    mapa = [[0] * largura for _ in range(altura)]

    # Adicionar obstáculos aleatórios
    for _ in range(100):  # Número de obstáculos
        x, y = random.randint(0, largura - 1), random.randint(0, altura - 1)
        mapa[y][x] = 1

    return mapa