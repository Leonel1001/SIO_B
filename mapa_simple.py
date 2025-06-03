def criar_mapa():
    largura, altura = 20, 20
    mapa = [[0]*largura for _ in range(altura)]

    for y in [4, 8, 12, 16]:
        for x in range(largura):
            mapa[y][x] = 1
        mapa[y][2] = 0
        mapa[y][17] = 0
    return mapa