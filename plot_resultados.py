import pandas as pd
import matplotlib.pyplot as plt

# Carregar CSV
df = pd.read_csv("resultados_treino.csv")

# Plot da Recompensa
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(df["episodio"], df["recompensa"], label="Recompensa Total", color="blue")
plt.xlabel("Episódio")
plt.ylabel("Recompensa")
plt.title("Recompensa por Episódio")
plt.grid(True)

# Plot do Sucesso acumulado
sucessos_acumulados = df["sucesso"].cumsum()
plt.subplot(1, 2, 2)
plt.plot(df["episodio"], sucessos_acumulados, label="Sucessos Acumulados", color="green")
plt.xlabel("Episódio")
plt.ylabel("Número de Sucessos")
plt.title("Sucessos Acumulados")
plt.grid(True)

plt.tight_layout()
plt.show()
