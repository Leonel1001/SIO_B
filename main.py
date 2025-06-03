import os
import pygame
from ffpyplayer.player import MediaPlayer
import numpy as np
from mapa_hard import criar_mapa_hard
from mapa_simple import criar_mapa
from algoritmo_astar import a_star
from simular import simular_isometrico_dqn
from simular_astar import animar_caminho_isometrico
from simular_ppo import simular_ppo
from simular_isometrico import simular_isometrico

pygame.init()

# Inicializa em fullscreen
infoObject = pygame.display.Info()
WINDOW_WIDTH = infoObject.current_w
WINDOW_HEIGHT = infoObject.current_h
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
pygame.display.set_caption("Simulação de Robôs Autónomos")

video_path = os.path.join("video", "6_65__Amazon Warehouse Order Picking Robots.mp4")

player = MediaPlayer(video_path, ff_opts={'an': True}) 

video_surface = None
clock = pygame.time.Clock()

def carregar_frame_video():
    global video_surface
    frame, val = player.get_frame()

    if val == 'eof':
        player.seek(0, relative=False)
        return

    if frame is None:
        return

    img, t = frame
    w, h = img.get_size()
    data = img.to_bytearray()[0]
    video_surface = pygame.image.frombuffer(data, (w, h), 'RGB')
    video_surface = pygame.transform.scale(video_surface, (WINDOW_WIDTH, WINDOW_HEIGHT))


def criar_mapa_aleatorio(tam=20, prob_obstaculo=0.2):
    mapa = np.random.choice([0, 1], size=(tam, tam), p=[1 - prob_obstaculo, prob_obstaculo])
    mapa[0][0] = 0
    return mapa.tolist()

def desenhar_logo_e_titulo(tela, font_titulo):
    logo = pygame.image.load("logo_menu_transparente.png").convert_alpha()
    logo = pygame.transform.scale(logo, (300, 200))
    rect_logo = logo.get_rect(center=(WINDOW_WIDTH // 2, 120))
    tela.blit(logo, rect_logo)

    titulo = font_titulo.render("Simulação de Robôs Autónomos", True, (255, 255, 255))
    sombra = font_titulo.render("Simulação de Robôs Autónomos", True, (0, 0, 0))
    texto_x = WINDOW_WIDTH // 2 - titulo.get_width() // 2
    tela.blit(sombra, (texto_x + 2, rect_logo.bottom + 8))
    tela.blit(titulo, (texto_x, rect_logo.bottom + 6))

def escolher_algoritmo():
    font = pygame.font.SysFont(None, 36)
    font_titulo = pygame.font.SysFont("arialblack", 32)
    clock = pygame.time.Clock()

    botoes = [
        ("A* ", animar_caminho_isometrico),
        ("DQN ", simular_isometrico_dqn),
        ("PPO ", simular_isometrico)
    ]

    while True:
        carregar_frame_video()
        if video_surface:
            screen.blit(video_surface, (0, 0))
        desenhar_logo_e_titulo(screen, font_titulo)

        mouse = pygame.mouse.get_pos()
        start_y = 400
        for i, (nome, _) in enumerate(botoes):
            rect = pygame.Rect(WINDOW_WIDTH // 2 - 150, start_y + i * 70, 400, 40)
            cor = (180, 220, 255) if rect.collidepoint(mouse) else (220, 220, 220)
            pygame.draw.rect(screen, cor, rect, border_radius=8)
            texto_botao = font.render(nome, True, (0, 0, 0))
            screen.blit(texto_botao, (rect.x + 10, rect.y + 5))

        voltar_rect = pygame.Rect(WINDOW_WIDTH // 2 - 70, start_y + 3 * 70, 140, 35)
        pygame.draw.rect(screen, (200, 100, 100), voltar_rect, border_radius=8)
        screen.blit(font.render("Voltar", True, (255, 255, 255)), (voltar_rect.x + 30, voltar_rect.y + 5))

        pygame.display.flip()
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                for i, (nome, func) in enumerate(botoes):
                    y_btn = start_y + i * 70
                    if WINDOW_WIDTH // 2 - 150 <= x <= WINDOW_WIDTH // 2 + 150 and y_btn <= y <= y_btn + 40:
                        return func
                if voltar_rect.collidepoint((x, y)):
                    return None

def selecionar_mapa_e_destino():
    font = pygame.font.SysFont(None, 28)
    font_titulo = pygame.font.SysFont("arialblack", 32)
    clock = pygame.time.Clock()

    mapas = {
        "Simples": criar_mapa,
        "Complexo": criar_mapa_hard,
        "Aleatório": criar_mapa_aleatorio
    }

    botoes = list(mapas.keys())
    mapa_selecionado = None
    destino_selecionado = None

    while True:
        carregar_frame_video()
        if video_surface:
            screen.blit(video_surface, (0, 0))
        desenhar_logo_e_titulo(screen, font_titulo)

        if not mapa_selecionado:
            texto = font.render("Selecione o tipo de mapa:", True, (255, 255, 255))
            screen.blit(texto, (WINDOW_WIDTH // 2 - texto.get_width() // 2, 220))
            mouse = pygame.mouse.get_pos()
            for i, nome in enumerate(botoes):
                y = 270 + i * 70
                rect = pygame.Rect(WINDOW_WIDTH // 2 - 150, y, 300, 50)
                cor = (180, 220, 255) if rect.collidepoint(mouse) else (200, 200, 200)
                pygame.draw.rect(screen, cor, rect, border_radius=8)
                texto_btn = font.render(nome, True, (0, 0, 0))
                screen.blit(texto_btn, (rect.x + 10, rect.y + 10))
        else:
            TAM = 20
            mapa_linhas = len(mapa_selecionado)
            mapa_colunas = len(mapa_selecionado[0])
            offset_x = (WINDOW_WIDTH - mapa_colunas * TAM) // 2
            offset_y = 100

            texto = font.render("Clique numa célula para definir o objetivo:", True, (255, 255, 255))
            screen.blit(texto, (WINDOW_WIDTH // 2 - texto.get_width() // 2, 30))

            for y in range(mapa_linhas):
                for x in range(mapa_colunas):
                    rect = pygame.Rect(offset_x + x * TAM, offset_y + y * TAM, TAM, TAM)
                    cor = (255, 255, 255) if mapa_selecionado[y][x] == 0 else (100, 100, 100)
                    pygame.draw.rect(screen, cor, rect)
                    pygame.draw.rect(screen, (150, 150, 150), rect, 1)

            if destino_selecionado:
                y, x = destino_selecionado
                rect = pygame.Rect(offset_x + x * TAM, offset_y + y * TAM, TAM, TAM)
                pygame.draw.rect(screen, (0, 255, 0), rect)

            continuar_rect = pygame.Rect(WINDOW_WIDTH // 2 - 100, 700, 200, 40)
            pygame.draw.rect(screen, (0, 180, 0), continuar_rect, border_radius=8)
            screen.blit(font.render("Continuar", True, (255, 255, 255)), (continuar_rect.x + 40, continuar_rect.y + 8))

            voltar_rect = pygame.Rect(30, 540, 120, 40)
            pygame.draw.rect(screen, (180, 50, 50), voltar_rect, border_radius=8)
            screen.blit(font.render("Voltar", True, (255, 255, 255)), (voltar_rect.x + 25, voltar_rect.y + 8))

        pygame.display.flip()
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                if not mapa_selecionado:
                    for i, nome in enumerate(botoes):
                        y_btn = 270 + i * 70
                        if WINDOW_WIDTH // 2 - 150 <= x <= WINDOW_WIDTH // 2 + 150 and y_btn <= y <= y_btn + 50:
                            mapa_selecionado = mapas[nome]()
                else:
                    TAM = 20
                    mapa_colunas = len(mapa_selecionado[0])
                    offset_x = (WINDOW_WIDTH - mapa_colunas * TAM) // 2
                    offset_y = 100
                    grid_x = (x - offset_x) // TAM
                    grid_y = (y - offset_y) // TAM
                    if 0 <= grid_y < len(mapa_selecionado) and 0 <= grid_x < len(mapa_selecionado[0]):
                        if mapa_selecionado[grid_y][grid_x] == 0:
                            destino_selecionado = (grid_y, grid_x)
                    continuar_rect = pygame.Rect(WINDOW_WIDTH // 2 - 100, 700, 200, 40)
                    voltar_rect = pygame.Rect(30, 540, 120, 40)
                    if continuar_rect.collidepoint((x, y)) and destino_selecionado:
                        return mapa_selecionado, destino_selecionado
                    elif voltar_rect.collidepoint((x, y)):
                        return None, None

def simular_com_esc(funcao, mapa, destino):
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return
        funcao(mapa, destino)
        running = False

if __name__ == "__main__":
    while True:
        mapa, destino = selecionar_mapa_e_destino()
        if mapa and destino:
            funcao = escolher_algoritmo()
            if funcao == animar_caminho_isometrico:
                caminho = a_star(mapa, (0, 0), destino)
                if caminho and isinstance(caminho, list):
                    simular_com_esc(lambda *_: animar_caminho_isometrico(mapa, caminho), mapa, destino)
                else:
                    print("Não foi possível encontrar caminho.")
            elif funcao:
                simular_com_esc(funcao, mapa, destino)
