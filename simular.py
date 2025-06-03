import pygame
import torch
import torch.nn as nn
import numpy as np
import time
from ambiente import Ambiente
from render_isometrico import desenhar_mapa_isometrico, desenhar_entidades

class SimuladorIsometrico:
    def __init__(self, mapa, objetivo, velocidade=5):
        self.mapa = mapa
        self.objetivo = objetivo
        self.velocidade = velocidade

        pygame.init()
        self.tela = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        pygame.display.set_caption("Simulação DQN Isométrica")
        self.clock = pygame.time.Clock()
        self.fonte = pygame.font.SysFont("Arial", 24)

        self.env = Ambiente(mapa, objetivo)
        self.estado = self.env.reset()
        self.ultima_acao = 0
        self.rastros = []
        self.stats = {"passos": 0, "recompensa": 0}
        self.paused = False
        self.running = True
        self.colisao = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.modelo = self.carregar_modelo()

        self.zoom = 1.0
        self.botao_zoom_in = pygame.Rect(20, 20, 40, 40)
        self.botao_zoom_out = pygame.Rect(70, 20, 40, 40)

        self.imagens = {
            "chao": pygame.image.load("assets_iso/chao_iso.png"),
            "obstaculo": pygame.image.load("assets_iso/obstaculo_iso.png"),
            "destino": pygame.image.load("assets_iso/destino_iso.png"),
            "robo": pygame.image.load("assets_iso/robo_iso.png")
        }

    def carregar_modelo(self):
        class DQN(nn.Module):
            def __init__(self):
                super(DQN, self).__init__()
                self.net = nn.Sequential(
                    nn.Linear(18, 512),
                    nn.ReLU(),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 4)
                )

            def forward(self, x):
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                return self.net(x)

        modelo = DQN().to(self.device)
        modelo.load_state_dict(torch.load("modelo_que_vai_funcionar.pth", map_location=self.device))
        modelo.eval()
        return modelo

    def escolher_acao(self, estado):
        acao_one_hot = np.zeros(4, dtype=np.float32)
        acao_one_hot[self.ultima_acao] = 1.0
        estado_completo = np.concatenate([estado, acao_one_hot])

        t = torch.FloatTensor(estado_completo).unsqueeze(0).to(self.device)
        with torch.no_grad():
            qs = self.modelo(t)
        return torch.argmax(qs).item()

    def desenhar_botoes_zoom(self):
        pygame.draw.rect(self.tela, (200, 200, 200), self.botao_zoom_in)
        pygame.draw.rect(self.tela, (200, 200, 200), self.botao_zoom_out)

        texto_mais = self.fonte.render("+", True, (0, 0, 0))
        texto_menos = self.fonte.render("-", True, (0, 0, 0))
        self.tela.blit(texto_mais, (self.botao_zoom_in.x + 12, self.botao_zoom_in.y + 5))
        self.tela.blit(texto_menos, (self.botao_zoom_out.x + 15, self.botao_zoom_out.y + 5))

    def desenhar(self):
        self.tela.fill((30, 30, 30))
        desenhar_mapa_isometrico(self.tela, self.mapa, self.imagens, self.zoom)
        desenhar_entidades(self.tela, self.env.estado, self.objetivo, self.imagens, self.zoom)
        self.desenhar_botoes_zoom()
        pygame.display.flip()

    def tratar_eventos(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    self.paused = not self.paused
                elif event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key in (pygame.K_PLUS, pygame.K_KP_PLUS):
                    self.velocidade += 5
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self.velocidade = max(1, self.velocidade - 5)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if self.botao_zoom_in.collidepoint(event.pos):
                    self.zoom = min(self.zoom * 1.1, 5.0)
                elif self.botao_zoom_out.collidepoint(event.pos):
                    self.zoom = max(self.zoom / 1.1, 0.3)

    def executar(self):
        while self.running:
            self.clock.tick(self.velocidade)
            self.tratar_eventos()

            if self.paused:
                continue

            acao = self.escolher_acao(self.estado)
            self.ultima_acao = acao
            resultado = self.env.step(acao)

            if len(resultado) == 4:
                proximo_estado, recompensa, feito, self.colisao = resultado
            else:
                proximo_estado, recompensa, feito = resultado
                self.colisao = recompensa <= -10

            self.rastros.append(self.env.estado)
            self.estado = proximo_estado
            self.stats["passos"] += 1
            self.stats["recompensa"] += recompensa

            self.desenhar()

            if feito:
                print("Encomenda encontrada!")
                time.sleep(2)
                self.running = False


def simular_isometrico_dqn(mapa, objetivo, velocidade=5):
    simulador = SimuladorIsometrico(mapa, objetivo, velocidade)
    simulador.executar()
