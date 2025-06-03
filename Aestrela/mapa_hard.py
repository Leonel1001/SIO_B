def criar_mapa_hard():
    largura, altura = 30, 30 

    mapa = [[0]*largura for _ in range(altura)]

    for y in range(2, altura - 2, 4):
        for x in range(1, largura - 1):
            if x % 5 != 0: 
                mapa[y][x] = 1

    for x in range(3, largura - 3, 6):
        for y in range(1, altura - 1):
            if y % 7 != 0:  
                mapa[y][x] = 1

    return mapa
