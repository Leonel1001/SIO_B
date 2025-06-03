import pygame
import time
from stable_baselines3 import PPO
from gym_env.ambiente_gym import CustomAmbiente

def simular_ppo(mapa, destino, modelo_path="ppo_agente_hard_dinamico"):
    pygame.init()

    # Tamanho e offsets
    tam_celula = 20
    offset_x, offset_y = 50, 130

    # Inicializa
    tela = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    largura, altura = tela.get_size()
    pygame.display.set_caption("Simulação PPO")

    clock = pygame.time.Clock()

    # Carregar imagens
    imagem_robo = pygame.transform.scale(pygame.image.load("assets/robo.png"), (20, 20))
    imagem_destino = pygame.transform.scale(pygame.image.load("assets/destino.png"), (20, 20))
    imagem_obstaculo = pygame.transform.scale(pygame.image.load("assets/obstaculo.png"), (20, 20))
    imagem_chao = pygame.transform.scale(pygame.image.load("assets/chao.png"), (20, 20))
    imagem_chegada = pygame.transform.scale(pygame.image.load("assets/chegada.png"), (20, 20))

    # Criar ambiente e carregar modelo
    env = CustomAmbiente(mapa, destino)
    model = PPO.load(modelo_path)
    estado, _ = env.reset()

    done = False
    chegada_frames = 0

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return

        # Ação do modelo
        acao, _ = model.predict(estado)
        estado, recompensa, done, _, _ = env.step(acao)

        # Desenhar mapa
        tela.fill((240, 240, 240))
        for y in range(len(mapa)):
            for x in range(len(mapa[0])):
                rect = pygame.Rect(offset_x + x * tam_celula, offset_y + y * tam_celula, tam_celula, tam_celula)
                if mapa[y][x] == 1:
                    tela.blit(imagem_obstaculo, rect)
                else:
                    tela.blit(imagem_chao, rect)
                pygame.draw.rect(tela, (100, 100, 100), rect, 1)

        # Destino
        y_d, x_d = destino
        rect_dest = pygame.Rect(offset_x + x_d * tam_celula, offset_y + y_d * tam_celula, tam_celula, tam_celula)
        if chegada_frames % 30 < 15:
            tela.blit(imagem_chegada, rect_dest)
        else:
            tela.blit(imagem_destino, rect_dest)
        chegada_frames += 1

        # Robô
        y_r, x_r = env.env.pos_agente
        rect_robo = pygame.Rect(offset_x + x_r * tam_celula, offset_y + y_r * tam_celula, tam_celula, tam_celula)
        tela.blit(imagem_robo, rect_robo)

        pygame.display.flip()
        clock.tick(10)
        time.sleep(0.05)

    time.sleep(1.5)
    pygame.quit()
