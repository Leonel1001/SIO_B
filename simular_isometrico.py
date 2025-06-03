import pygame
import time
from render_isometrico import desenhar_mapa_isometrico, desenhar_entidades
from stable_baselines3 import PPO
from gym_env.ambiente_gym import CustomAmbiente

def simular_isometrico(mapa, destino, modelo_path="ppo_agente_hard_dinamico"):
    pygame.init()

    tela = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    largura, altura = tela.get_size()
    pygame.display.set_caption("Simulação Isométrica PPO")

    clock = pygame.time.Clock()
    fonte = pygame.font.SysFont("Arial", 24)

    imagens = {
        "chao": pygame.image.load("assets_iso/chao_iso.png"),
        "obstaculo": pygame.image.load("assets_iso/obstaculo_iso.png"),
        "destino": pygame.image.load("assets_iso/destino_iso.png"),
        "robo": pygame.image.load("assets_iso/robo_iso.png")
    }

    zoom = 1.0
    botao_zoom_in = pygame.Rect(20, 20, 40, 40)
    botao_zoom_out = pygame.Rect(70, 20, 40, 40)

    env = CustomAmbiente(mapa, destino)
    model = PPO.load(modelo_path)
    estado, _ = env.reset()
    done = False

    while not done:
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
        desenhar_entidades(tela, env.env.pos_agente, destino, imagens, zoom)

        # Desenhar botões
        pygame.draw.rect(tela, (200, 200, 200), botao_zoom_in)
        pygame.draw.rect(tela, (200, 200, 200), botao_zoom_out)
        tela.blit(fonte.render("+", True, (0, 0, 0)), (botao_zoom_in.x + 12, botao_zoom_in.y + 5))
        tela.blit(fonte.render("-", True, (0, 0, 0)), (botao_zoom_out.x + 15, botao_zoom_out.y + 5))

        pygame.display.flip()
        clock.tick(8)
        time.sleep(0.1)

        acao, _ = model.predict(estado)
        estado, _, done, _, _ = env.step(acao)

    time.sleep(1.5)
