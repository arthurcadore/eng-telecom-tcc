# %%
# Importando as bibliotecas necessárias
import numpy as np
import komm as komm
import matplotlib.pyplot as plt
import scienceplots
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

# Imprimindo a versão da biblioteca
print("Komm version: ", komm.__version__)
print("Numpy version: ", np.__version__)

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
# Quantidade de blocos: 
h = 200

# Polinômios geradores
polinomios_geradores = [[0b1110011, 0b1101101]]

# Ordem da modulação PSK
M = 4

# Número de quadros
Nframes = 30000

# Faixa de valores de Eb/N0, com passo 1 dB
# EBN0_Range = np.arange(-1, 7, 1)

# NOTA: Neste ponto foi utilizado um passo de 0.5 dB para melhorar a resolução do gráfico
EBN0_Range = np.arange(-1, 10, 0.5)

# %%
# Número de processadores para paralelismo
N_PROCS = os.cpu_count()  # ou defina manualmente, ex: N_PROCS = 4

# Função para processar cada Eb/N0 em paralelo

def process_ebn0(args):
    i, SNR, h, Nframes, polinomios_geradores, M = args
    import numpy as np
    import komm
    # Recria os objetos em cada processo
    modulator = komm.PSKModulation(M, labeling='reflected')
    conv_encoder = komm.ConvolutionalCode(feedforward_polynomials=polinomios_geradores)
    block_code = komm.TerminatedConvolutionalCode(conv_encoder, h, mode='zero-termination')
    encoder = komm.BlockEncoder(block_code)
    decoder = komm.BlockDecoder(block_code, method='viterbi_hard')
    bits = np.random.randint(0, 2, h * Nframes)
    encoded_bits = encoder(bits)
    modulated_bits = modulator.modulate(encoded_bits)
    modulated_bits_original = modulator.modulate(bits)
    snr = float(10 ** (SNR / 10))
    awgn = komm.AWGNChannel(snr=snr, signal_power="measured")
    noisy_signal = awgn(modulated_bits)
    noisy_signal_original = awgn(modulated_bits_original)
    demodulated_signal = modulator.demodulate(noisy_signal)
    demodulated_signal_original = modulator.demodulate(noisy_signal_original)
    decoded_bits = decoder(demodulated_signal)
    BER_conv = np.mean(bits != decoded_bits[:len(bits)])
    BER_original = np.mean(bits != demodulated_signal_original[:len(bits)])
    return i, BER_conv, BER_original

# Inicializando o vetor de BER para transmissão sem codificação
BER_original = np.zeros(len(EBN0_Range))
# Inicializando o vetor de BER para o código convolucional
BER_conv = np.zeros(len(EBN0_Range))

# Monta lista de argumentos para cada processo
args_list = [(i, SNR, h, Nframes, polinomios_geradores, M) for i, SNR in enumerate(EBN0_Range)]

# Processamento paralelo com barra de progresso
with ProcessPoolExecutor(max_workers=N_PROCS) as executor:
    futures = [executor.submit(process_ebn0, args) for args in args_list]
    for f in tqdm(as_completed(futures), total=len(futures), desc='Paralelo', unit='ponto'):
        i, ber_c, ber_o = f.result()
        BER_conv[i] = ber_c
        BER_original[i] = ber_o

print("BER_conv     | BER_original")
for conv, original in zip(BER_conv, BER_original):
    print(f"{conv:<12} | {original}")

# %%
fig, ax = plt.subplots(figsize=(16, 9))

ax.semilogy(EBN0_Range, BER_conv, '-o', label='Codificado Convolucional', 
            color="k", markerfacecolor='white', markersize=6, linewidth=2)
ax.semilogy(EBN0_Range, BER_original, '-s', label='Não Codificado', 
            color="blue", markerfacecolor='white', markersize=6, linewidth=2)

ax.set_xlabel('$E_b/N_0$ (dB)', fontsize=14)
ax.set_ylabel('Bit Error Rate (BER)', fontsize=14)
ax.set_title('Desempenho BER vs $E_b/N_0$ para QPSK com e sem codificação convolucional', fontsize=16)

# Define a grade em todos os múltiplos de 0.5 dB
ax.set_xticks(np.arange(-1, 11, 0.5), minor=True)  # Grid secundário
ax.set_xticks(np.arange(-1, 11, 1))                # Ticks principais com rótulo
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)

# define o tamanho da fonte dos rótulos dos eixos
ax.tick_params(axis='x', labelsize=22)
ax.tick_params(axis='y', labelsize=22)

# define o tamanho da fonte do título dos eixos
ax.set_title('Desempenho BER vs $E_b/N_0$ para QPSK com e sem codificação convolucional', fontsize=22)

# define o tamanho da fonte do rótulo do eixo x
ax.set_xlabel('$E_b/N_0$ (dB)', fontsize=22)

# define o tamanho da fonte do rótulo do eixo y
ax.set_ylabel('Bit Error Rate (BER)', fontsize=22)


# Grade visível para ticks principais e secundários
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

ax.set_ylim(1e-4, 1)

ax.legend(
    loc='upper right',
    frameon=True,
    edgecolor='black',
    facecolor='white',
    fontsize=12,
    fancybox=True
)

# %%
plt.tight_layout()
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "out")
os.makedirs(output_dir, exist_ok=True)
output_pdf = os.path.join(output_dir, "convolutional.pdf")
fig.savefig(output_pdf, dpi=1500, bbox_inches='tight', format='pdf')