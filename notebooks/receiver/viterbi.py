# %%
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
import scienceplots
import numpy as np
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from tqdm import tqdm  # em vez de from tqdm.notebook import tqdm
import os


plt.style.use('science')
plt.rcParams["figure.figsize"] = (16, 9)

plt.rc('font', size=16)          # tamanho da fonte geral (eixos, ticks)
plt.rc('axes', titlesize=22)     # tamanho da fonte do título dos eixos
plt.rc('axes', labelsize=22)     # tamanho da fonte dos rótulos dos eixos (xlabel, ylabel)
plt.rc('xtick', labelsize=16)    # tamanho da fonte dos ticks no eixo x
plt.rc('ytick', labelsize=16)    # tamanho da fonte dos ticks no eixo y
plt.rc('legend', fontsize=16)    # tamanho da fonte das legendas (se houver)
plt.rc('figure', titlesize=22)   # tamanho da fonte do título da figura (plt.suptitle)

# %%
# Geradores binários (G0 = 171 octal = 1111001 bin, G1 = 133 octal = 1011011 bin)
G0 = [1, 1, 1, 0, 0, 1, 1]
G1 = [1, 1, 0, 1, 1, 0, 1]

def convolucao(u, g):
    return sum([bit * gen for bit, gen in zip(u, g)]) % 2

def get_output(state, input_bit):
    u = [input_bit] + [int(b) for b in f"{state:06b}"]
    v0 = convolucao(u, G0)
    v1 = convolucao(u, G1)
    return f"{v0}{v1}"

def get_next_state(state, input_bit):
    bits = f"{state:06b}"
    new_state_bits = f"{input_bit}{bits[:5]}"
    return int(new_state_bits, 2)

# Criar treliça completa com todos os 64 estados (2^6) e 128 transições (2^7)
all_states = list(range(64))  # Todos os estados de 0 a 63 (registrador de 6 bits)
transitions = []

# Gerar todas as transições possíveis
print("Gerando transições da treliça...")
for state in tqdm(all_states, desc="Estados"):
    for input_bit in [0, 1]:
        next_state = get_next_state(state, input_bit)
        output = get_output(state, input_bit)
        transitions.append((state, next_state, input_bit, output))

# Criar grafo com networkx
G = nx.DiGraph()

# Adicionar todos os nós
for state in all_states:
    G.add_node(f"{state:06b}")

# Adicionar todas as arestas
print("Adicionando arestas ao grafo...")
for from_state, to_state, input_bit, output in tqdm(transitions, desc="Arestas"):
    G.add_edge(f"{from_state:06b}", f"{to_state:06b}", label=f"{input_bit}/{output}", input_bit=input_bit)

# Plotar treliça com layout shell (mais adequado para visualizar a estrutura da treliça)
print("Calculando layout da treliça...")
pos = nx.shell_layout(G)
plt.figure(figsize=(20, 20))  # Figura maior para acomodar todos os estados

# Separar arestas por bit de entrada
edges_0 = [(u, v) for (u, v, d) in G.edges(data=True) if d['input_bit'] == 0]
edges_1 = [(u, v) for (u, v, d) in G.edges(data=True) if d['input_bit'] == 1]

# Desenhar nós (todos na mesma cor)
nx.draw_networkx_nodes(G, pos, node_color='lightblue', edgecolors='black', 
                      node_size=800)

# Desenhar arestas com cores diferentes
nx.draw_networkx_edges(G, pos, edgelist=edges_0, edge_color='blue', 
                      width=2.5, arrows=True, arrowstyle='->', arrowsize=15, 
                      label='Bit de saída (input=0)')
nx.draw_networkx_edges(G, pos, edgelist=edges_1, edge_color='red', 
                      width=2.5, arrows=True, arrowstyle='->', arrowsize=15, 
                      label='Bit de saída (input=1)')

# Desenhar labels dos nós
nx.draw_networkx_labels(G, pos, font_size=8)

# Adicionar legenda explicativa com cores
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='blue', lw=3, label='Transição com output=0'),
    Line2D([0], [0], color='red', lw=3, label='Transição com output=1')
]

plt.legend(
        handles=legend_elements,
        title='Cores das Transições',
        loc='upper right',
        frameon=True,
        edgecolor='black',
        facecolor='white',
        fontsize=12,
        fancybox=True
    )
          

plt.title("Trelha Completa do Codificador Convolucional (64 Estados, 128 Transições)")
plt.axis('off')

# Salvar PDF na pasta ./output
print("Salvando arquivo...")
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "out")
os.makedirs(output_dir, exist_ok=True)
output_pdf = os.path.join(output_dir, "viterbi_trelha_completa.pdf")
plt.savefig(output_pdf, dpi=300, bbox_inches='tight', format='pdf')
plt.close()
print(f"✓ Salvo: {output_pdf}")

print(f"\n✅ Trelha completa gerada com {len(list(G.nodes()))} estados e {len(list(G.edges()))} transições")
print(f"📁 Arquivo salvo em: {output_dir}")
print("🎉 Processo concluído!")