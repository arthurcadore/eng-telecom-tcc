# %%
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import os
import matplotlib.patches as mpatches
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
# Parâmetros
n_bits = 63
amostras_por_bit = 10
total_amostras = n_bits * amostras_por_bit
t = np.linspace(0, n_bits, total_amostras)


# %%
# Função para gerar vetores I e Q com 1000 bits
def gerar_vetores_IQ(n_bits=1000):
    I = np.random.randint(0, 2, n_bits)
    Q = np.random.randint(0, 2, n_bits)
    return I, Q

# %%
def codificar_nrz(bits, amostras_por_bit=10):
    nrz = []
    for bit in bits:
        nivel = 1 if bit == 1 else -1
        nrz.extend([nivel] * amostras_por_bit)
    return np.array(nrz)

# %%
# Codificação Manchester para vetor Q
def codificar_manchester(bits, amostras_por_bit=10):
    manchester = []
    meio = amostras_por_bit // 2
    for bit in bits:
        if bit == 1:
            manchester.extend([1] * meio + [-1] * (amostras_por_bit - meio))
        else:
            manchester.extend([-1] * meio + [1] * (amostras_por_bit - meio))
    return np.array(manchester)

# %%
I, Q = gerar_vetores_IQ(n_bits)
sinal_I_nrz = codificar_nrz(I, amostras_por_bit)
sinal_Q_manchester = codificar_manchester(Q, amostras_por_bit)

print("Vetor I:", I)
print("Vetor Q:", Q)

print("Sinal I (NRZ):", len(sinal_I_nrz), "amostras")
print("Sinal Q (Manchester):", len(sinal_Q_manchester), "amostras")

# %%
# Função para garantir diretório de saída relativo ao script
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, '../../out')
os.makedirs(output_dir, exist_ok=True)

# %%
# Plotar sinais codificados
def plotar_codificacoes(t, sinal_nrz, sinal_manchester, n_bits=1000, amostras_por_bit=10):
    fig, axs = plt.subplots(2, 1, sharex=True)
   
    axs[0].plot(t, sinal_nrz, drawstyle='steps-post', color='k', linewidth=2)
    axs[0].set_title("Canal I - Codificação NRZ")
    axs[0].set_ylabel("Amplitude")
    axs[0].grid(True)
    
    axs[1].plot(t, sinal_manchester, drawstyle='steps-post', color='k', linewidth=2)
    axs[1].set_title("Canal Q - Codificação Manchester")
    axs[1].set_ylabel("Amplitude")
    axs[1].set_xlabel("Tempo (amostras)")
    axs[1].grid(True)

    plt.tight_layout()
    pdf_path = os.path.join(output_dir, 'modulador_codificacoes.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=1500)
    plt.close(fig)

plotar_codificacoes(t, sinal_I_nrz, sinal_Q_manchester)


# %%
# Eixo de tempo: mapeia 630 amostras para 63 bits
t = np.linspace(0, len(I)-1, len(sinal_I_nrz))

# Fase QPSK: mapeamento (I_bit, Q_bit) -> fase
qpsk_phase_map = {
    (0, 0): -3*np.pi/4,
    (0, 1): +3*np.pi/4,
    (1, 0): -1*np.pi/4,
    (1, 1): +1*np.pi/4
}
phases = [qpsk_phase_map[(i, q)] for i, q in zip(I, Q)]

# Tempo dos símbolos QPSK (um por bit)
t_phase = np.arange(len(phases))

# Criação dos subplots
fig, axs = plt.subplots(2, 1, figsize=(16, 9), sharex=True)

# Subplot 1: sinais I e Q
axs[0].step(t, sinal_I_nrz, where='post', label='I (NRZ)', linewidth=2, color='k')
axs[0].step(t, sinal_Q_manchester, where='post', label='Q (Manchester)', linestyle='--', linewidth=2, color='b')
axs[0].set_ylim([-1.5, 1.5])
axs[0].set_ylabel("Amplitude")
axs[0].set_title("Sinais Codificados - I e Q")
axs[0].legend(
    loc='upper right',
    frameon=True,
    edgecolor='black',     # Cor da borda
    facecolor='white',     # Cor de fundo da caixa
    fontsize=12,
    fancybox=True          # Cantos arredondados
)

axs[0].grid(True)

axs[1].step(t_phase, phases, where='post', color='k', linewidth=2, label='Fase QPSK')
axs[1].set_yticks([ -3*np.pi/4, -np.pi/4, np.pi/4, 3*np.pi/4 ])
axs[1].set_yticklabels([
    r"$S_0 = -\frac{3\pi}{4}$",
    r"$S_2 = -\frac{\pi}{4}$",
    r"$S_3 = +\frac{\pi}{4}$",
    r"$S_1 = +\frac{3\pi}{4}$"
])
axs[1].set_ylabel("Fase")
axs[1].set_xlabel("Tempo (em bits)")
axs[1].set_title("Símbolos QPSK")
axs[1].grid(True)
axs[1].legend(
    loc='upper right',
    frameon=True,
    edgecolor='black',     # Cor da borda
    facecolor='white',     # Cor de fundo da caixa
    fontsize=12,
    fancybox=True          # Cantos arredondados
)


plt.tight_layout()
pdf_path2 = os.path.join(output_dir, 'modulador_sinais_fase.pdf')
plt.savefig(pdf_path2, format='pdf', bbox_inches='tight', dpi=1500)
plt.close(fig)

# %%
# Mapeamento de bits (I, Q) para símbolos QPSK complexos (Gray coding)
bit_pairs = {
    (0, 0): (1 + 1j),
    (0, 1): (-1 + 1j),
    (1, 1): (-1 - 1j),
    (1, 0): (1 - 1j),
}

# Criar vetor de símbolos complexos a partir dos bits I e Q
symbols_qpsk = np.array([bit_pairs[(i, q)] for i, q in zip(I, Q)]) / np.sqrt(2)

print("Símbolos QPSK complexos:", symbols_qpsk)


# %%
# Separar parte real e imaginária
i_signal = np.real(symbols_qpsk)
q_signal = np.imag(symbols_qpsk)

# Tempo simulado (um ponto por símbolo)
t = np.arange(len(symbols_qpsk))


# Constelação QPSK (Plano I/Q) e sinais no tempo
fig2, axs2 = plt.subplots(1, 2, figsize=(16, 9))

# Subplot 1: Constelação
axs2[0].plot(i_signal, q_signal, 'o')
axs2[0].axhline(0, color='gray', lw=0.5)
axs2[0].axvline(0, color='gray', lw=0.5)
axs2[0].grid(True)
axs2[0].set_title('Constelação QPSK (Plano I/Q)')
axs2[0].set_xlabel('In-Phase (I)')
axs2[0].set_ylabel('Quadrature (Q)')
axs2[0].set_xlim(-1.2, 1.2)
axs2[0].set_ylim(-1.2, 1.2)

# Subplot 2: Sinal QPSK no tempo
axs2[1].stem(t, i_signal, linefmt='b-', markerfmt='bo', basefmt=" ", label='I (In-Phase)')
axs2[1].stem(t, q_signal, linefmt='r-', markerfmt='ro', basefmt=" ", label='Q (Quadrature)')
axs2[1].grid(True)
axs2[1].set_title('Sinal QPSK no Tempo')
axs2[1].set_xlabel('Símbolos (S)')
axs2[1].set_ylabel('Amplitude')
axs2[1].legend()

fig2.tight_layout()
pdf_path3 = os.path.join(output_dir, 'modulador_constelacao_tempo.pdf')
fig2.savefig(pdf_path3, format='pdf', bbox_inches='tight', dpi=1500)
plt.close(fig2)

# Exemplo clássico QPSK: os símbolos (complexos) podem ser definidos assim (ordem correta):
symbols_qpsk_classica = np.array([
    -1/np.sqrt(2) - 1j/np.sqrt(2),   # S0, bits 00, fase -3pi/4
    -1/np.sqrt(2) + 1j/np.sqrt(2),   # S1, bits 01, fase +3pi/4
    1/np.sqrt(2) - 1j/np.sqrt(2),    # S2, bits 10, fase -pi/4
    1/np.sqrt(2) + 1j/np.sqrt(2)     # S3, bits 11, fase +pi/4
])
phases_str = [r"$-\frac{3\pi}{4}$", r"$+\frac{3\pi}{4}$", r"$-\frac{\pi}{4}$", r"$+\frac{\pi}{4}$"]

# Parâmetros para o sinal QPSK no tempo
samples_per_symbol = 100       # Amostras por símbolo
fc = 2                         # Frequência da portadora (Hz)
symbol_rate = 1                # 1 símbolo por segundo
fs = samples_per_symbol * symbol_rate  # Taxa de amostragem
num_symbols = len(symbols_qpsk_classica)        # Tamanho real do vetor de símbolos
t = np.arange(0, num_symbols, 1/fs)    # Tempo total

# Fases dos símbolos (ângulo dos pontos complexos)
phases = np.angle(symbols_qpsk_classica)

# Gerar o sinal QPSK no tempo (modulação por fase)
carrier_signal = np.zeros_like(t)
for i in range(num_symbols):
    t_start = i * samples_per_symbol
    t_end = (i + 1) * samples_per_symbol
    carrier_signal[t_start:t_end] = np.cos(2 * np.pi * fc * t[t_start:t_end] + phases[i])

# Constelação clássica QPSK (subplot 1) e Sinal QPSK no tempo (subplot 2)
fig4, axs4 = plt.subplots(1, 2, figsize=(18, 9))

# Subplot 1: Constelação clássica
circle = mpatches.Circle((0, 0), 1, color='gray', fill=False, linestyle='--', linewidth=2)
axs4[0].add_patch(circle)
axs4[0].plot(np.real(symbols_qpsk_classica), np.imag(symbols_qpsk_classica), 'ro', markersize=10)
for s, phase_text in zip(symbols_qpsk_classica, phases_str):
    axs4[0].plot([0, np.real(s)], [0, np.imag(s)], 'k--', linewidth=2)
    axs4[0].text(np.real(s)+0.10, np.imag(s)+0.01, phase_text, fontsize=16, color='k')
axs4[0].axhline(0, color='gray', lw=0.5)
axs4[0].axvline(0, color='gray', lw=0.5)
axs4[0].grid(True)
axs4[0].set_title('Constelação QPSK (Plano I/Q)')
axs4[0].set_xlabel('In-Phase (I)')
axs4[0].set_ylabel('Quadrature (Q)')
axs4[0].set_xlim(-1.2, 1.2)
axs4[0].set_ylim(-1.2, 1.2)

# Subplot 2: Sinal QPSK no tempo
axs4[1].plot(t, carrier_signal, 'k', linewidth=2)
axs4[1].set_title('Sinal QPSK no tempo')
axs4[1].set_xlabel('Tempo (s)')
axs4[1].set_ylabel('Amplitude')
axs4[1].set_xlim(0, min(4, t[-1]))
axs4[1].grid(True)

fig4.tight_layout()
pdf_path5 = os.path.join(output_dir, 'modulador_constelacao_classica_com_tempo.pdf')
fig4.savefig(pdf_path5, format='pdf', bbox_inches='tight', dpi=1500)
plt.close(fig4)



