import pygame

def desenhar_mapa_isometrico(tela, mapa, imagens, zoom=1.0):
    base_tile_width = 64
    base_tile_height = 32
    tile_width = int(base_tile_width * zoom)
    tile_height = int(base_tile_height * zoom)

    largura_mapa = len(mapa[0])
    altura_mapa = len(mapa)

    offset_x = tela.get_width() // 2 - (tile_width // 2)
    offset_y = 50 

    for y in range(altura_mapa):
        for x in range(largura_mapa):
            tipo = mapa[y][x]
            imagem_base = imagens["chao"] if tipo == 0 else imagens["obstaculo"]
            tile = pygame.transform.scale(imagem_base, (tile_width, tile_height))

            iso_x = (x - y) * (tile_width // 2) + offset_x
            iso_y = (x + y) * (tile_height // 2) + offset_y

            tela.blit(tile, (iso_x, iso_y))

def desenhar_entidades(tela, pos_robo, pos_destino, imagens, zoom=1.0):
    base_tile_width = 64
    base_tile_height = 32
    tile_width = int(base_tile_width * zoom)
    tile_height = int(base_tile_height * zoom)

    offset_x = tela.get_width() // 2 - (tile_width // 2)
    offset_y = 50

    for nome, (y, x) in [("destino", pos_destino), ("robo", pos_robo)]:
        imagem_base = imagens[nome]
        imagem = pygame.transform.scale(imagem_base, (tile_width, tile_height))

        iso_x = (x - y) * (tile_width // 2) + offset_x
        iso_y = (x + y) * (tile_height // 2) + offset_y

        tela.blit(imagem, (iso_x, iso_y))
