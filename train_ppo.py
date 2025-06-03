import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from gym_env.ambient_hard_dinamico import CustomAmbienteHardDinamico
from mapa_hard import criar_mapa_hard

# Criar mapa complexo e destino válido
mapa = criar_mapa_hard()
while True:
    destino = (np.random.randint(0, len(mapa)), np.random.randint(0, len(mapa[0])))
    if mapa[destino[0]][destino[1]] == 0:
        break

# Criar ambiente

env = CustomAmbienteHardDinamico()
check_env(env, warn=True)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=600_000)
model.save("ppo_agente_hard_dinamico")

print("Continuação do treino PPO em mapa_hard concluída.")
