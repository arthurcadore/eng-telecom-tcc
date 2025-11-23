# %%
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
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
fs = 10e6          # taxa de amostragem: 10 MHz
fc = 400e3         # frequência da portadora: 400 kHz
duration = 0.082   # duração: 82 ms
SNR_dB = 20        # relação sinal-ruído

N = int(duration * fs)
t = np.arange(N) / fs

# %%
carrier = np.cos(2 * np.pi * fc * t)
signal_power = np.mean(carrier**2)
noise_power = signal_power / (10**(SNR_dB / 10))
noise1 = np.sqrt(noise_power) * np.random.randn(N)
carrier_awgn = carrier + noise1
fft_carrier = np.fft.fft(carrier_awgn)


# %%
bit_rate = 400
symbol_rate = bit_rate / 2
samples_per_symbol = int(fs / symbol_rate)

num_bits = int(duration * bit_rate)
if num_bits % 2 != 0:
    num_bits += 1
num_symbols = num_bits // 2

bits = np.random.randint(0, 2, num_bits)
mapping = {
    (0,0): (1+1j)/np.sqrt(2),
    (0,1): (-1+1j)/np.sqrt(2),
    (1,1): (-1-1j)/np.sqrt(2),
    (1,0): (1-1j)/np.sqrt(2)
}
symbols = np.array([mapping[tuple(bits[2*i:2*i+2])] for i in range(num_symbols)])
baseband = np.zeros(N, dtype=complex)
for i, sym in enumerate(symbols):
    start = i * samples_per_symbol
    end = start + samples_per_symbol
    if end > N:
        break
    baseband[start:end] = sym

I = np.real(baseband)
Q = np.imag(baseband)
modulated = I * np.cos(2 * np.pi * fc * t) - Q * np.sin(2 * np.pi * fc * t)

signal_power_mod = np.mean(modulated**2)
noise_power_mod = signal_power_mod / (10**(SNR_dB / 10))
noise2 = np.sqrt(noise_power_mod) * np.random.randn(N)
modulated_awgn = modulated + noise2
fft_mod = np.fft.fft(modulated_awgn)


# %%
f = np.fft.fftfreq(N, d=1/fs)
f_mask = (f >= fc - 10e3) & (f <= fc + 10e3)
f_abs_khz = f[f_mask] / 1e3  # frequência absoluta em kHz

mag_carrier_db = 20 * np.log10(np.abs(fft_carrier[f_mask]) / N)
mag_modulated_db = 20 * np.log10(np.abs(fft_mod[f_mask]) / N)

mag_min = -100
mag_max = max(mag_carrier_db.max(), mag_modulated_db.max())


# %%
plt.figure(figsize=(16, 8))
plt.style.use('science')

plt.subplot(1, 2, 1)
plt.plot(f_abs_khz, mag_carrier_db, color='navy', linewidth=1.2, label='Portadora pura')
plt.axhline(y=-20, color='red', linestyle='--', linewidth=1, label='Limiar de detecção')
plt.title('Portadora pura')
plt.xlabel('Frequência (kHz)')
plt.ylabel('Magnitude (dB)')
plt.grid(True)
plt.xticks(np.arange(390, 411, 5))  # passo de 1 kHz
plt.ylim(mag_min, mag_max)
plt.legend(
    loc='upper right',
    frameon=True,
    edgecolor='black',     # Cor da borda
    facecolor='white',     # Cor de fundo da caixa
    fontsize=12,
    fancybox=True          # Cantos arredondados
)
# plt.legend(['Poradora Pura'], loc='upper right', frameon=True, edgecolor='black', facecolor='white', fontsize=12, fancybox=True)


plt.subplot(1, 2, 2)
plt.plot(f_abs_khz, mag_modulated_db, color='darkgreen', linewidth=1.2, label='Sinal modulado')
plt.axhline(y=-20, color='red', linestyle='--', linewidth=1, label='Limiar de detecção')
plt.title('Sinal Modulado (400 bps)')
plt.xlabel('Frequência (kHz)')
plt.grid(True)
plt.xticks(np.arange(390, 411, 5))  # passo de 1 kHz
plt.ylim(mag_min, mag_max)
plt.legend(
    
    loc='upper right',
    frameon=True,
    edgecolor='black',     # Cor da borda
    facecolor='white',     # Cor de fundo da caixa
    fontsize=12,
    fancybox=True          # Cantos arredondados
)
# plt.legend(['Sinal Modulado'], loc='upper right', frameon=True, edgecolor='black', facecolor='white', fontsize=12, fancybox=True)

plt.tight_layout()
# Garantir diretório de saída relativo ao script
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, '..', '..', 'out')
os.makedirs(output_dir, exist_ok=True)
pdf_path = os.path.join(output_dir, 'carrier_spectra.pdf')
plt.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=1500)



