import torch
import numpy as np
import random
from ambiente import Ambiente
from mapa_simple import criar_mapa
from mapa_hard import criar_mapa_hard
from mapa_dinamico import criar_mapa_dinamico

class DQN(torch.nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(14, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 4),
        )

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x)

# Configurações
N_TESTES = 300
PASSOS_MAX = 500
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carrega o modelo treinado
modelo = DQN().to(device)
modelo.load_state_dict(torch.load("modelo_final_final.pth", map_location=device))
modelo.eval()

def escolher_acao(estado):
    with torch.no_grad():
        t = torch.FloatTensor(estado).unsqueeze(0).to(device)
        qs = modelo(t)
    return torch.argmax(qs).item()

# Validação em vários mapas
sucessos = 0
mapas = [criar_mapa, criar_mapa_hard, criar_mapa_dinamico]

for i in range(N_TESTES):
    mapa = random.choice(mapas)()
    while True:
        destino = (random.randint(0, len(mapa) - 1), random.randint(0, len(mapa[0]) - 1))
        if mapa[destino[0]][destino[1]] == 0:
            break

    env = Ambiente(mapa, destino)
    estado = env.reset()

    for _ in range(PASSOS_MAX):
        acao = escolher_acao(estado)
        estado, recompensa, done, _ = env.step(acao)
        if done and recompensa >= 100:
            sucessos += 1
            break

print(f"Taxa de sucesso: {sucessos}/{N_TESTES} = {sucessos / N_TESTES:.2%}")
