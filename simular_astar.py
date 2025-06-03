import pygame
import time
from render_isometrico import desenhar_mapa_isometrico, desenhar_entidades

def animar_caminho_isometrico(mapa, caminho):
    # Inicializar pygame
    if not pygame.get_init():
        pygame.init()

    # Modo fullscreen
    tela = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    largura_tela, altura_tela = tela.get_size()
    pygame.display.set_caption("Execução do caminho A* (Isométrico)")

    clock = pygame.time.Clock()
    fonte = pygame.font.SysFont("Arial", 24)

    # Carregar imagens isométricas
    imagens = {
        "chao": pygame.image.load("assets_iso/chao_iso.png").convert_alpha(),
        "obstaculo": pygame.image.load("assets_iso/obstaculo_iso.png").convert_alpha(),
        "destino": pygame.image.load("assets_iso/destino_iso.png").convert_alpha(),
        "robo": pygame.image.load("assets_iso/robo_iso.png").convert_alpha()
    }

    destino = caminho[-1]
    zoom = 1.0

    # Definir botões de zoom
    botao_zoom_in = pygame.Rect(20, 20, 40, 40)
    botao_zoom_out = pygame.Rect(70, 20, 40, 40)

    for i, (y, x) in enumerate(caminho):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if botao_zoom_in.collidepoint(event.pos):
                    zoom = min(zoom * 1.1, 5.0)
                elif botao_zoom_out.collidepoint(event.pos):
                    zoom = max(zoom / 1.1, 0.3)

        tela.fill((30, 30, 30))
        desenhar_mapa_isometrico(tela, mapa, imagens, zoom)
        desenhar_entidades(tela, (y, x), destino, imagens, zoom)

        # Desenhar botões de zoom
        pygame.draw.rect(tela, (200, 200, 200), botao_zoom_in)
        pygame.draw.rect(tela, (200, 200, 200), botao_zoom_out)
        tela.blit(fonte.render("+", True, (0, 0, 0)), (botao_zoom_in.x + 12, botao_zoom_in.y + 5))
        tela.blit(fonte.render("-", True, (0, 0, 0)), (botao_zoom_out.x + 15, botao_zoom_out.y + 5))

        pygame.display.flip()
        clock.tick(10)
        time.sleep(0.1) 

    time.sleep(2)  
