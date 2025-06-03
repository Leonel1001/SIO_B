import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from mapa_hard import criar_mapa_hard
from mapa_simple import criar_mapa
from mapa_dinamico import criar_mapa_dinamico
from ambiente import Ambiente
import matplotlib.pyplot as plt

plt.ion()
fig, ax = plt.subplots()
(linha_recompensa,) = ax.plot([], [], label="Recompensa por episódio")
ax.set_xlabel("Episódio")
ax.set_ylabel("Recompensa Total")
ax.set_title("Fine-Tuning do Agente DQN")
ax.grid(True)
ax.legend()

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x)

# Hiperparâmetros de fine-tuning
gamma = 0.98
epsilon = 0.3  # exploração inicial
epsilon_decay = 0.995
batch_size = 64
memory_size = 50000
episodes = 8000
learning_rate = 0.0003
target_update_freq = 10
passos_por_episodio = 500

# Ambiente e rede
input_dim = 14
output_dim = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy_net = DQN(input_dim, output_dim).to(device)
target_net = DQN(input_dim, output_dim).to(device)
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

policy_net.load_state_dict(torch.load("modelo_final_final.pth"))
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
memory = deque(maxlen=memory_size)

melhor_recompensa = -float("inf")
recompensas_totais = []

# Função auxiliar: escolher ação
def escolher_acao(estado, epsilon):
    if random.random() < epsilon:
        return random.randint(0, output_dim - 1)
    with torch.no_grad():
        estado_t = torch.FloatTensor(estado).to(device)
        q_vals = policy_net(estado_t)
    return torch.argmax(q_vals).item()

# Função auxiliar: otimizar modelo
def otimizar_modelo():
    if len(memory) < batch_size:
        return
    batch = random.sample(memory, batch_size)
    estados, acoes, recompensas, prox_estados, dones = zip(*batch)

    estados = torch.FloatTensor(np.array(estados)).to(device)
    acoes = torch.LongTensor(acoes).unsqueeze(1).to(device)
    recompensas = torch.FloatTensor(recompensas).to(device)
    prox_estados = torch.FloatTensor(np.array(prox_estados)).to(device)
    dones = torch.FloatTensor(dones).to(device)

    q_atual = policy_net(estados).gather(1, acoes).squeeze()
    with torch.no_grad():
        q_alvo = recompensas + gamma * target_net(prox_estados).max(1)[0] * (1 - dones)

    loss = nn.MSELoss()(q_atual, q_alvo)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    optimizer.step()

# Loop principal de treino
for episodio in range(episodes):
    if episodio > 100:
        epsilon = max(0.05, epsilon * epsilon_decay)

    mapa_base = random.choice([criar_mapa(), criar_mapa_hard(), criar_mapa_dinamico()])

    objetivo_valido = False
    while not objetivo_valido:
        destino = (random.randint(0, len(mapa_base) - 1), random.randint(0, len(mapa_base[0]) - 1))
        if mapa_base[destino[0]][destino[1]] == 0:
            objetivo_valido = True

    env = Ambiente(mapa_base, destino)
    estado = env.reset()
    total_recompensa = 0
    visitados = set()
    visitados.add(tuple(env.pos_agente))
    dist_anterior = np.linalg.norm(np.array(env.pos_agente) - np.array(env.objetivo))
    repeticoes = 0

    for _ in range(passos_por_episodio):
        pos_tuple = tuple(env.pos_agente)
        acao = escolher_acao(estado, epsilon)
        prox_estado, recompensa, done, _ = env.step(acao)

        if pos_tuple in visitados:
            recompensa -= 5
            repeticoes += 1
        else:
            visitados.add(pos_tuple)
            repeticoes = 0

        if done and recompensa >= 50:
            recompensa = 500
        elif done:
            recompensa = -100
        else:
            dist_atual = np.linalg.norm(np.array(env.pos_agente) - np.array(env.objetivo))
            delta_dist = dist_anterior - dist_atual
            recompensa -= 1
            recompensa += delta_dist * 2
            dist_anterior = dist_atual

            if repeticoes >= 10:
                recompensa -= 10

        memory.append((estado, acao, recompensa, prox_estado, done))
        estado = prox_estado
        total_recompensa += recompensa

        if done:
            break

        otimizar_modelo()

    recompensas_totais.append(total_recompensa)

    if total_recompensa > melhor_recompensa:
        melhor_recompensa = total_recompensa
        torch.save(policy_net.state_dict(), "modelo_melhor_finetune.pth")

    if episodio % target_update_freq == 0:
        target_net.load_state_dict(policy_net.state_dict())

    if episodio % 25 == 0:
        media_ultimos = np.mean(recompensas_totais[-100:]) if len(recompensas_totais) >= 100 else np.mean(recompensas_totais)
        print(f"[FineTune] Episódio {episodio}: Recompensa = {total_recompensa:.1f} | Média últimos 100 = {media_ultimos:.1f}")
        linha_recompensa.set_data(range(len(recompensas_totais)), recompensas_totais)
        ax.relim()
        ax.autoscale_view()

    torch.save(policy_net.state_dict(), "modelo_temp_finetune.pth")

plt.ioff()
plt.show()
torch.save(policy_net.state_dict(), "modelo_novo.pth")
