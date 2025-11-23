import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import butter, filtfilt, freqz, impulse, lfilter

import scienceplots
import os

def mag2db(signal):
    mag = np.abs(signal)
    mag /= np.max(mag)
    return 20 * np.log10(mag + 1e-12)  # evita log(0)

# Estilo visual
plt.style.use('science')
plt.rcParams["figure.figsize"] = (16, 9)
plt.rc('font', size=16)
plt.rc('axes', titlesize=22)
plt.rc('axes', labelsize=22)
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.rc('legend', fontsize=16)
plt.rc('figure', titlesize=22)

# estilo da legenda 
# Configuração global via rc
plt.rc('legend',
       frameon=True,
       edgecolor='black',
       facecolor='white',
       fancybox=True,
       fontsize=12)

# 1. Vetores binários de entrada
Ie = np.random.randint(0, 2, 20)
Qe = np.random.randint(0, 2, 20)

# 2. Codificação NRZ (Ie -> Xnrz)
Xnrz = np.repeat(Ie, 2)

# 3. Codificação Manchester (Qe -> Ym)
Ym = np.empty(Qe.size * 2, dtype=int)
Ym[::2] = 1 - Qe
Ym[1::2] = Qe

# 4. Superamostragem dos vetores de entrada
Ie_up = np.repeat(Ie, 2)
Qe_up = np.repeat(Qe, 2)

# 5. Eixo de tempo compartilhado
x = np.arange(len(Ie_up))  # 40 posições

# 6. Plot
fig, axs = plt.subplots(4, 1, sharex=True)

# Posições de grade: uma linha vertical entre cada bit (0, 2, ..., 40)
bit_edges = np.arange(0, len(Ie_up)+1, 2)

def setup_grid(ax):
    ax.set_xlim(0, len(Ie_up))
    ax.set_ylim(-0.2, 1.4)
    ax.grid(False)  # desativa grid padrão
    for pos in bit_edges:
        ax.axvline(x=pos, color='gray', linestyle='--', linewidth=0.5)

# Ie original com bits sobrepostos (no topo, com margem)
axs[0].step(x, Ie_up, where='post', label=r"Canal I $(X_n)$", color='blue', linewidth=2)
for i, bit in enumerate(Ie):
    axs[0].text(i*2 + 1, 1.15, str(bit), ha='center', va='bottom', fontsize=14)
axs[0].set_ylabel(r"$X_n$")
axs[0].legend(loc='upper right')
setup_grid(axs[0])

leg0 = axs[0].legend(
    loc='upper right', frameon=True, edgecolor='black',
    facecolor='white', fontsize=12, fancybox=True
)
leg0.get_frame().set_facecolor('white')
leg0.get_frame().set_edgecolor('black')
leg0.get_frame().set_alpha(1.0)


# Ie codificado NRZ com pares de bits (00 ou 11)
axs[1].step(x, Xnrz, where='post', label=r"I codificado $(NRZ)$", color='navy', linewidth=2)
for i in range(len(Ie)):
    pair = ''.join(str(b) for b in Xnrz[2*i:2*i+2])
    axs[1].text(i*2 + 1, 1.15, pair, ha='center', va='bottom', fontsize=14)
axs[1].set_ylabel(r"$X_{NRZ}[n]$")
axs[1].legend(loc='upper right')
setup_grid(axs[1])

leg1 = axs[1].legend(
    loc='upper right', frameon=True, edgecolor='black',
    facecolor='white', fontsize=12, fancybox=True
)
leg1.get_frame().set_facecolor('white')
leg1.get_frame().set_edgecolor('black')
leg1.get_frame().set_alpha(1.0)


# Qe original com bits sobrepostos (no topo, com margem)
axs[2].step(x, Qe_up, where='post', label=r"Canal Q $(Y_n)$", color='green', linewidth=2)
for i, bit in enumerate(Qe):
    axs[2].text(i*2 + 1, 1.15, str(bit), ha='center', va='bottom', fontsize=14)
axs[2].set_ylabel(r"$Y_n$")
axs[2].legend(loc='upper right')
setup_grid(axs[2])

leg2 = axs[2].legend(
    loc='upper right', frameon=True, edgecolor='black',
    facecolor='white', fontsize=12, fancybox=True
)
leg2.get_frame().set_facecolor('white')
leg2.get_frame().set_edgecolor('black')
leg2.get_frame().set_alpha(1.0)

# Qe codificado Manchester com pares de bits (01 ou 10)
axs[3].step(x, Ym, where='post', label=r"Q codificado $(Manchester)$", color='darkgreen', linewidth=2)
for i in range(len(Qe)):
    pair = ''.join(str(b) for b in Ym[2*i:2*i+2])
    axs[3].text(i*2 + 1, 1.15, pair, ha='center', va='bottom', fontsize=14)
axs[3].set_ylabel(r"$Y_M[n]$")
axs[3].legend(loc='upper right')
leg3 = axs[3].legend(
    loc='upper right', frameon=True, edgecolor='black',
    facecolor='white', fontsize=12, fancybox=True
)
leg3.get_frame().set_facecolor('white')
leg3.get_frame().set_edgecolor('black')
leg3.get_frame().set_alpha(1.0)
setup_grid(axs[3])

plt.xlabel('Amostras')
plt.tight_layout()
plt.subplots_adjust(top=0.92)

# 7. Salvar como PDF em ../out/
output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'out')
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'codificacao_iq.pdf'))


# 5. Eixo de tempo
x = np.arange(len(Ie_up))
bit_edges = np.arange(0, len(Ie_up) + 1, 2)

# 6. Função de grid com ylim customizável
def setup_grid2(ax, ylim=(-0.2, 1.2)):
    ax.set_ylim(*ylim)
    ax.grid(False)
    for pos in bit_edges:
        ax.axvline(x=pos, color='gray', linestyle='--', linewidth=0.5)

# 7. Mapeamento QPSK corrigido — baseado nos bits a cada amostra
Ie_bitwise = np.repeat(Ie, 2)

phase_map = {
    (0, 0): -3 * np.pi / 4,
    (0, 1):  3 * np.pi / 4,
    (1, 0): -1 * np.pi / 4,
    (1, 1):  1 * np.pi / 4
}

phases_up = np.array([
    phase_map[(Ie_bitwise[i], Ym[i])] for i in range(len(Ym))
])

# 8. Plot com 3 subplots
fig2, axs2 = plt.subplots(3, 1, sharex=True, figsize=(16, 9))
fig2.suptitle('Codificação e Mapeamento em Fase QPSK')

# Subplot 1: Ie codificado NRZ
axs2[0].step(x, Xnrz, where='post', label='Ie codificado (NRZ)', color='navy', linewidth=2)
axs2[0].set_ylabel('NRZ')
setup_grid2(axs2[0], ylim=(-0.2, 1.2))
axs2[0].legend(loc='upper right')
leg0 = axs2[0].legend(
    loc='upper right', frameon=True, edgecolor='black',
    facecolor='white', fontsize=12, fancybox=True
)
leg0.get_frame().set_facecolor('white')
leg0.get_frame().set_edgecolor('black')
leg0.get_frame().set_alpha(1.0)

# Subplot 2: Qe codificado Manchester
axs2[1].step(x, Ym, where='post', label='Qe codificado (Manchester)', color='darkgreen', linewidth=2)
axs2[1].set_ylabel('Manchester')
setup_grid2(axs2[1], ylim=(-0.2, 1.2))
axs2[1].legend(loc='upper right')

leg1 = axs2[1].legend(
    loc='upper right', frameon=True, edgecolor='black',
    facecolor='white', fontsize=12, fancybox=True
)
leg1.get_frame().set_facecolor('white')
leg1.get_frame().set_edgecolor('black')
leg1.get_frame().set_alpha(1.0)

# Subplot 3: Fase QPSK
axs2[2].step(x, phases_up, where='post', label='Fase QPSK (rad)', color='purple', linewidth=2)
axs2[2].set_ylabel('Fase (rad)')
axs2[2].set_yticks([-3 * np.pi / 4, -np.pi / 4, np.pi / 4, 3 * np.pi / 4])
axs2[2].set_yticklabels([
    r'$-\frac{3\pi}{4}$',
    r'$-\frac{\pi}{4}$',
    r'$\frac{\pi}{4}$',
    r'$\frac{3\pi}{4}$'
])
axs2[2].legend(loc='upper right')
leg2 = axs2[2].legend(
    loc='upper right', frameon=True, edgecolor='black',
    facecolor='white', fontsize=12, fancybox=True
)
leg2.get_frame().set_facecolor('white')
leg2.get_frame().set_edgecolor('black')
leg2.get_frame().set_alpha(1.0)
setup_grid2(axs2[2])
axs2[2].set_ylim(-3*np.pi/4 - 0.2, 3*np.pi/4 + 0.2)

axs2[2].set_xlabel('Amostras')

# 9. Ajuste do limite X para cobrir todo o comprimento dos degraus e alinhar todos os gráficos
xlim_val = (0, len(Ie_up))
for ax in axs2:
    ax.set_xlim(*xlim_val)

plt.tight_layout()
plt.subplots_adjust(top=0.92, bottom=0.08, hspace=0.3)

# 10. Salvar como PDF
output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'out')
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'fase_qpsk_codificacao.pdf'))

# plt.show()  # Descomente para visualizar interativamente


fs = 128_000  # taxa de amostragem em Hz
Rb = 400      # taxa de bits em bps
Tb = 1 / Rb   # duração de um bit
sps = int(fs / Rb)  # samples per symbol: 320


# 11. Pulso RRC (Raised Cosine Limited)
def rcc_pulse(t, Tb, alpha):
    num = np.sinc(t / Tb)
    den = 1 - (2 * alpha * t / Tb) ** 2
    with np.errstate(divide='ignore', invalid='ignore'):
        rc = num * np.cos(np.pi * alpha * t / Tb) / den
        rc[np.isnan(rc)] = 0
        rc[np.isinf(rc)] = 0
    return rc

alpha = 0.8
span = 6  # duração total do pulso: 6*T
t_rc = np.linspace(-span * Tb, span * Tb, span * sps * 2)
g = rcc_pulse(t_rc, Tb, alpha)

# 12. Interpolação dos sinais Ie e Qe (NRZ e Manchester)
def interpolate(symbols, pulse, sps):
    upsampled = np.zeros(len(symbols) * sps)
    upsampled[::sps] = symbols
    return np.convolve(upsampled, pulse, mode='same')

# Reutiliza os símbolos Xnrz e Ym
d_I = interpolate(Xnrz, g, sps)
d_Q = interpolate(Ym, g, sps)

# Eixo de tempo comum
t_interp = np.arange(len(d_I)) / fs

# 13. Layout com 3 plots (1 grande acima, 2 abaixo lado a lado)
fig_interp = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])

# Subplot superior (0, 0) ocupando as duas colunas
ax_rcc = fig_interp.add_subplot(gs[0, :])
ax_rcc.plot(t_rc, g, label=r'Pulso RRC ($\alpha=0.8$)', color='red', linewidth=2)
ax_rcc.set_title('Pulso Root Raised Cosine (RRC)')
ax_rcc.set_ylabel('Amplitude')
ax_rcc.grid(True)
ax_rcc.legend()

leg_rcc = ax_rcc.legend(
    loc='upper right', frameon=True, edgecolor='black',
    facecolor='white', fontsize=12, fancybox=True
)
leg_rcc.get_frame().set_facecolor('white')
leg_rcc.get_frame().set_edgecolor('black')
leg_rcc.get_frame().set_alpha(1.0)

# Subplot inferior esquerdo: I(t)
ax_I = fig_interp.add_subplot(gs[1, 0])
ax_I.plot(t_interp, d_I, label=r"$dI(t)$ - NRZ", color='navy', linewidth=2)
ax_I.set_title(r"Sinal $dI(t)$ - Formatação com RRC")
ax_I.set_xlabel('Tempo (s)')
ax_I.set_ylabel('Amplitude')
ax_I.set_xlim(0, 0.05)
ax_I.grid(True)
ax_I.legend()

leg_I = ax_I.legend(
    loc='upper right', frameon=True, edgecolor='black',
    facecolor='white', fontsize=12, fancybox=True
)
leg_I.get_frame().set_facecolor('white')
leg_I.get_frame().set_edgecolor('black')
leg_I.get_frame().set_alpha(1.0)

# Subplot inferior direito: Q(t)
ax_Q = fig_interp.add_subplot(gs[1, 1])
ax_Q.plot(t_interp, d_Q, label=r"$dQ(t)$ - Manchester", color='darkgreen', linewidth=2)
ax_Q.set_title(r"Sinal $dQ(t)$ - Formatação com RRC")
ax_Q.set_xlabel('Tempo (s)')
ax_Q.set_ylabel('Amplitude')
ax_Q.set_xlim(0, 0.05)
ax_Q.grid(True)
ax_Q.legend()

leg_Q = ax_Q.legend(
    loc='upper right', frameon=True, edgecolor='black',
    facecolor='white', fontsize=12, fancybox=True
)
leg_Q.get_frame().set_facecolor('white')
leg_Q.get_frame().set_edgecolor('black')
leg_Q.get_frame().set_alpha(1.0)

# Layout
plt.tight_layout()
plt.subplots_adjust(top=0.92, hspace=0.4)
plt.savefig(os.path.join(output_dir, 'pulso_rrc_e_sinais_iq.pdf'))

# 14. Modulação QPSK com portadora de 4kHz
fc = 2_000  # Hz
cos_carrier = np.cos(2 * np.pi * fc * t_interp)
sin_carrier = np.sin(2 * np.pi * fc * t_interp)
s_mod = d_I * cos_carrier - d_Q * sin_carrier  # QPSK: I*cos - Q*sin

# 15. Mapeamento para constelação discreta
I_symbols = 2 * Ie - 1  # 0 → -1, 1 → +1
Q_symbols = 2 * Qe - 1

# 16. Figura com layout 2x2 (mas constelação ocupa duas linhas)
fig_all = plt.figure(figsize=(16, 9))
gs_all = gridspec.GridSpec(2, 2, height_ratios=[1, 1])

# (0,0) Sinais interpolados
ax_interp = fig_all.add_subplot(gs_all[0, 0])
ax_interp.plot(t_interp, d_I, label=r'$dI(t)$ - NRZ', color='navy', linewidth=2)
ax_interp.plot(t_interp, d_Q, label=r'$dQ(t)$ - Manchester', color='darkgreen', linewidth=2)
ax_interp.set_title(r'Sinais Interpolados $dI(t)$ e $dQ(t)$')
ax_interp.set_ylabel('Amplitude')
ax_interp.set_xlim(0, 0.05)
ax_interp.grid(True)
ax_interp.legend(loc='upper right')

leg_Interp = ax_interp.legend(
    loc='upper right', frameon=True, edgecolor='black',
    facecolor='white', fontsize=12, fancybox=True
)
leg_Interp.get_frame().set_facecolor('white')
leg_Interp.get_frame().set_edgecolor('black')
leg_Interp.get_frame().set_alpha(1.0)

# (1,0) Sinal modulado
ax_mod = fig_all.add_subplot(gs_all[1, 0])
ax_mod.plot(t_interp, s_mod, label=r'Sinal QPSK Modulado', color='purple', linewidth=1)
ax_mod.set_title(r'Sinal QPSK Modulado ($f_c$ = $2kHz$)')
ax_mod.set_xlabel('Tempo (s)')
ax_mod.set_ylabel('Amplitude')
ax_mod.set_xlim(0, 0.05)
ax_mod.grid(True)
ax_mod.legend(loc='upper right')

leg_mod = ax_mod.legend(
    loc='upper right', frameon=True, edgecolor='black',
    facecolor='white', fontsize=12, fancybox=True
)
leg_mod.get_frame().set_facecolor('white')
leg_mod.get_frame().set_edgecolor('black')
leg_mod.get_frame().set_alpha(1.0)

# (0:2, 1) Constelação QPSK ocupa as duas linhas da direita
ax_const = fig_all.add_subplot(gs_all[:, 1])
ax_const.plot(I_symbols, Q_symbols, 'o', color='k', markersize=12)

labels = ['s0', 's1', 's2', 's3']

# Extrai pontos únicos da constelação
unique_points = list(set(zip(I_symbols, Q_symbols)))
unique_points.sort()  # para manter uma ordem estável

for i, (i_val, q_val) in enumerate(unique_points):
    if i < len(labels):  # evita estouro
        ax_const.text(i_val + 0.05, q_val + 0.05, labels[i],
                      fontsize=18, color='black', ha='left', va='bottom')

ax_const.axhline(0, color='gray', linestyle='--')
ax_const.axvline(0, color='gray', linestyle='--')
ax_const.set_title('Constelação QPSK')
ax_const.set_xlabel('Canal I')
ax_const.set_ylabel('Canal Q')
ax_const.set_xlim(-1.5, 1.5)
ax_const.set_ylim(-1.5, 1.5)
ax_const.set_xticks([-1, 0, 1])
ax_const.set_yticks([-1, 0, 1])

ax_const.set_aspect('equal', adjustable='box')
ax_const.grid(True)

# Layout final
plt.tight_layout()
plt.subplots_adjust(hspace=0.35, wspace=0.25)
plt.savefig(os.path.join(output_dir, 'qpsk_sinais_e_constelacao.pdf'))


# 17. Cálculo da envoltória
envoltoria = np.sqrt(d_I**2 + d_Q**2)

# 18. Plot da envoltória do sinal QPSK
fig_env = plt.figure(figsize=(16, 5))
plt.plot(t_interp, envoltoria, color='orange', linewidth=2, label='Envoltória')
plt.title('Envoltória do Sinal QPSK Modulado')
plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude')
plt.xlim(0, 0.05)
plt.grid(True)
plt.legend()

# Salvar
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'envoltoria_qpsk.pdf'))


# 19. Adição de ruído AWGN com SNR = 10 dB
def add_awgn(signal, snr_db):
    signal_power = np.mean(signal**2)
    snr_linear = 10**(snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power) * np.random.randn(len(signal))
    return signal + noise

s_mod_ruido = add_awgn(s_mod, snr_db=10)

# 20. Plot 2x2 - Tempo e Frequência
fig_tf = plt.figure(figsize=(16, 10))
gs_tf = gridspec.GridSpec(2, 2)

# Tempo - Sem ruído
ax1 = fig_tf.add_subplot(gs_tf[0, 0])
ax1.plot(t_interp, s_mod, label='Sinal Sem Ruído', color='blue')
ax1.set_title('Tempo - Sem Ruído')
ax1.set_xlim(0, 0.01)
ax1.set_xlabel('Tempo (s)')
ax1.set_ylabel('Amplitude')
ax1.grid(True)
ax1.legend()

leg1 = ax1.legend(
    loc='upper right', frameon=True, edgecolor='black',
    facecolor='white', fontsize=12, fancybox=True
)
leg1.get_frame().set_facecolor('white')
leg1.get_frame().set_edgecolor('black')
leg1.get_frame().set_alpha(1.0)

# Tempo - Com ruído
ax2 = fig_tf.add_subplot(gs_tf[0, 1])
ax2.plot(t_interp, s_mod_ruido, label='Sinal com AWGN (10 dB)', color='red')
ax2.set_title('Tempo - Com Ruído (SNR = 10 dB)')
ax2.set_xlim(0, 0.01)
ax2.set_xlabel('Tempo (s)')
ax2.set_ylabel('Amplitude')
ax2.grid(True)
ax2.legend()

leg2 = ax2.legend(
    loc='upper right', frameon=True, edgecolor='black',
    facecolor='white', fontsize=12, fancybox=True
)
leg2.get_frame().set_facecolor('white')
leg2.get_frame().set_edgecolor('black')
leg2.get_frame().set_alpha(1.0)

# Frequência - Sem ruído
ax3 = fig_tf.add_subplot(gs_tf[1, 0])

fft_clean = np.fft.fftshift(np.fft.fft(s_mod))
freqs = np.fft.fftshift(np.fft.fftfreq(len(s_mod), d=1/fs))
fft_clean_db = mag2db(fft_clean)
ax3.plot(freqs, fft_clean_db, color='blue')
ax3.set_ylim(-60, 5)
ax3.set_xlim(-2.5 * fc, 2.5 * fc)
ax3.set_ylabel("Magnitude (dB)")
ax3.grid(True)

# Frequência - Com ruído
ax4 = fig_tf.add_subplot(gs_tf[1, 1])
fft_noisy = np.abs(np.fft.fftshift(np.fft.fft(s_mod_ruido)))
fft_noisy = np.fft.fftshift(np.fft.fft(s_mod_ruido))
fft_noisy_db = mag2db(fft_noisy)
ax4.plot(freqs, fft_noisy_db, color='red')
ax4.set_ylim(-60, 5)
ax4.set_xlim(-2.5 * fc, 2.5 * fc)
ax4.set_ylabel("Magnitude (dB)")
ax4.grid(True)

# Layout final
plt.tight_layout()
plt.subplots_adjust(hspace=0.3)
plt.savefig(os.path.join(output_dir, 'qpsk_ruido_10db_2x2.pdf'))


# 1. Multiplicação por portadoras (já foi feito acima, mas recriando por clareza)
cos_demod = 2 * np.cos(2 * np.pi * fc * t_interp)
sin_demod = 2 * np.sin(2 * np.pi * fc * t_interp)
y_I_ = s_mod_ruido * cos_demod
y_Q_ = s_mod_ruido * sin_demod


# === PLOT 1: FFT dos sinais y'_I(t) e y'_Q(t) ===
fig_fft_prod = plt.figure(figsize=(16, 8))
gs_fft = gridspec.GridSpec(2, 1)

# FFT de y_I_
YI_f = np.fft.fftshift(np.fft.fft(y_I_))
freqs = np.fft.fftshift(np.fft.fftfreq(len(y_I_), d=1/fs))
YI_db = mag2db(YI_f)

ax_fft_i = fig_fft_prod.add_subplot(gs_fft[0])
ax_fft_i.plot(freqs, YI_db, color='blue', label=r"$|X_I(f)|$")
ax_fft_i.set_xlim(-2.5 * fc, 2.5 * fc)
ax_fft_i.set_ylim(-60, 5)
ax_fft_i.set_title(r"Espectro do canal I - $x_I(t)$")
ax_fft_i.set_xlabel("Frequência (Hz)")
ax_fft_i.set_ylabel("Magnitude (dB)")
ax_fft_i.grid(True)
ax_fft_i.legend()
leg_fft_i = ax_fft_i.legend(
    loc='upper right', frameon=True, edgecolor='black',
    facecolor='white', fontsize=12, fancybox=True
)
leg_fft_i.get_frame().set_facecolor('white')
leg_fft_i.get_frame().set_edgecolor('black')
leg_fft_i.get_frame().set_alpha(1.0)

# FFT de y_Q_
YQ_f = np.fft.fftshift(np.fft.fft(y_Q_))
YQ_db = mag2db(YQ_f)

ax_fft_q = fig_fft_prod.add_subplot(gs_fft[1])
ax_fft_q.plot(freqs, YQ_db, color='green', label=r"$|Y_Q(f)|$")
ax_fft_q.set_xlim(-2.5 * fc, 2.5 * fc)
ax_fft_q.set_ylim(-60, 5)
ax_fft_q.set_title(r"Espectro do canal Q - $y_Q(t)$")
ax_fft_q.set_xlabel("Frequência (Hz)")
ax_fft_q.set_ylabel("Magnitude (dB)")
ax_fft_q.grid(True)
ax_fft_q.legend()
leg_fft_q = ax_fft_q.legend(
    loc='upper right', frameon=True, edgecolor='black',
    facecolor='white', fontsize=12, fancybox=True
)
leg_fft_q.get_frame().set_facecolor('white')
leg_fft_q.get_frame().set_edgecolor('black')
leg_fft_q.get_frame().set_alpha(1.0)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fft_produtos_com_portadoras.pdf'))





# === FILTRO ===
cutoff = 1500  # Hz
order = 6
b, a = butter(order, cutoff / (0.5 * fs), btype='low')
d_I_rec = filtfilt(b, a, y_I_)
d_Q_rec = filtfilt(b, a, y_Q_)

# === REMOÇÃO DE OFFSET DC ===
d_I_rec -= np.mean(d_I_rec)
d_Q_rec -= np.mean(d_Q_rec)

# === REESCALAMENTO DE AMPLITUDE ===
d_I_rec *= 2
d_Q_rec *= 2

# Resposta ao impulso
impulse_len = 512
impulse_input = np.zeros(impulse_len)
impulse_input[0] = 1  # impulso unitário
impulse_response = lfilter(b, a, impulse_input)

t_imp = np.arange(impulse_len) / fs

# === PLOT 2: Resposta ao impulso e sinais filtrados ===
fig_filt = plt.figure(figsize=(16, 10))
gs_filt = gridspec.GridSpec(3, 1)

# Resposta ao impulso
ax_imp = fig_filt.add_subplot(gs_filt[0])
ax_imp.plot(t_imp * 1000, impulse_response, color='red', label='Resposta ao Impulso - FPB', linewidth=2)
ax_imp.set_title("Resposta ao Impulso do Filtro Passa-Baixa")
ax_imp.set_xlabel("Tempo (ms)")
ax_imp.set_ylabel("Amplitude")
ax_imp.grid(True)
ax_imp.legend()
leg_imp = ax_imp.legend(
    loc='upper right', frameon=True, edgecolor='black',
    facecolor='white', fontsize=12, fancybox=True
)
leg_imp.get_frame().set_facecolor('white')
leg_imp.get_frame().set_edgecolor('black')
leg_imp.get_frame().set_alpha(1.0)

# I filtrado
ax_fi = fig_filt.add_subplot(gs_filt[1])
ax_fi.plot(t_interp, d_I_rec, color='blue', label=r"$d_I(t)$ filtrado")
ax_fi.set_title("Canal I após filtragem passa-baixa")
ax_fi.set_xlim(0, 0.1)
ax_fi.set_xlabel("Tempo (s)")
ax_fi.set_ylabel("Amplitude")
ax_fi.grid(True)
ax_fi.legend()
leg_fi = ax_fi.legend(
    loc='upper right', frameon=True, edgecolor='black',
    facecolor='white', fontsize=12, fancybox=True
)
leg_fi.get_frame().set_facecolor('white')
leg_fi.get_frame().set_edgecolor('black')
leg_fi.get_frame().set_alpha(1.0)

# Q filtrado
ax_fq = fig_filt.add_subplot(gs_filt[2])
ax_fq.plot(t_interp, d_Q_rec, color='green', label=r"$d_Q(t)$ filtrado")
ax_fq.set_title("Canal Q após filtragem passa-baixa")
ax_fq.set_xlim(0, 0.1)
ax_fq.set_xlabel("Tempo (s)")
ax_fq.set_ylabel("Amplitude")
ax_fq.grid(True)
ax_fq.legend()
leg_fq = ax_fq.legend(
    loc='upper right', frameon=True, edgecolor='black',
    facecolor='white', fontsize=12, fancybox=True
)
leg_fq.get_frame().set_facecolor('white')
leg_fq.get_frame().set_edgecolor('black')
leg_fq.get_frame().set_alpha(1.0)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'filtro_e_demodulados_filtrados.pdf'))



# === PLOT 3: Resposta ao impulso + espectros antes/depois da filtragem ===
fig_spec = plt.figure(figsize=(16, 10))
gs_spec = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])

# Linha 1 (0, :) - Resposta ao impulso
ax_impulse = fig_spec.add_subplot(gs_spec[0, :])
ax_impulse.plot(t_imp * 1000, impulse_response, color='red', linewidth=2, label='Resposta ao Impulso - FPB')
ax_impulse.set_title("Resposta ao Impulso do Filtro Passa-Baixa")
ax_impulse.set_xlabel("Tempo (ms)")
ax_impulse.set_ylabel("Amplitude")
ax_impulse.grid(True)
ax_impulse.legend()
leg_impulse = ax_impulse.legend(
    loc='upper right', frameon=True, edgecolor='black',
    facecolor='white', fontsize=12, fancybox=True
)
leg_impulse.get_frame().set_facecolor('white')
leg_impulse.get_frame().set_edgecolor('black')
leg_impulse.get_frame().set_alpha(1.0)

# FFT antes da filtragem (y'_I)
YI_f = np.fft.fftshift(np.fft.fft(y_I_))
freqs = np.fft.fftshift(np.fft.fftfreq(len(y_I_), d=1/fs))

YI_db = mag2db(YI_f)
ax_yi = fig_spec.add_subplot(gs_spec[1, 0])
ax_yi.plot(freqs, YI_db, color='blue', label=r"$|X'_I(f)|$")
ax_yi.set_xlim(-2.5 * fc, 2.5 * fc)
ax_yi.set_title(r"Espectro de $x'_I(t)$ (Antes do FPB)")
ax_yi.set_xlabel("Frequência (Hz)")
ax_yi.set_ylabel("Magnitude (dB)")
ax_yi.set_ylim(-80, 5)
ax_yi.grid(True)
ax_yi.legend()
leg_yi = ax_yi.legend(
    loc='upper right', frameon=True, edgecolor='black',
    facecolor='white', fontsize=12, fancybox=True
)
leg_yi.get_frame().set_facecolor('white')
leg_yi.get_frame().set_edgecolor('black')
leg_yi.get_frame().set_alpha(1.0)

# FFT depois da filtragem (d_I)
DI_f = np.fft.fftshift(np.fft.fft(d_I_rec))
DI_db = mag2db(DI_f)
ax_di = fig_spec.add_subplot(gs_spec[1, 1])
ax_di.plot(freqs, DI_db, color='darkblue', label=r"$|d'_I(f)|$")
ax_di.set_xlim(-2.5 * fc, 2.5 * fc)
ax_di.set_ylim(-80, 5)
ax_di.set_title(r"Espectro de $d'_I(t)$ (Após FPB)")
ax_di.set_xlabel("Frequência (Hz)")
ax_di.set_ylabel("Magnitude (dB)")
ax_di.grid(True)
ax_di.legend()
leg_di = ax_di.legend(
    loc='upper right', frameon=True, edgecolor='black',
    facecolor='white', fontsize=12, fancybox=True
)
leg_di.get_frame().set_facecolor('white')
leg_di.get_frame().set_edgecolor('black')
leg_di.get_frame().set_alpha(1.0)

# FFT antes da filtragem (y'_Q)
YQ_f = np.fft.fftshift(np.fft.fft(y_Q_))
YQ_db = mag2db(YQ_f)
ax_yq = fig_spec.add_subplot(gs_spec[2, 0])
ax_yq.plot(freqs, YQ_db, color='green', label=r"$|Y'_Q(f)|$")
ax_yq.set_xlim(-2.5 * fc, 2.5 * fc)
ax_yq.set_title(r"Espectro de $y'_Q(t)$ (Antes do FPB)")
ax_yq.set_xlabel("Frequência (Hz)")
ax_yq.set_ylabel("Magnitude (dB)")
ax_yq.set_ylim(-90, 5)
ax_yq.grid(True)
ax_yq.legend()
leg_yq = ax_yq.legend(
    loc='upper right', frameon=True, edgecolor='black',
    facecolor='white', fontsize=12, fancybox=True
)
leg_yq.get_frame().set_facecolor('white')
leg_yq.get_frame().set_edgecolor('black')
leg_yq.get_frame().set_alpha(1.0)

# FFT depois da filtragem (d_Q)
DQ_f = np.fft.fftshift(np.fft.fft(d_Q_rec))
DQ_db = mag2db(DQ_f)
ax_dq = fig_spec.add_subplot(gs_spec[2, 1])
ax_dq.plot(freqs, DQ_db, color='darkgreen', label=r"$|d_Q(f)|$")
ax_dq.set_xlim(-2.5 * fc, 2.5 * fc)
ax_dq.set_ylim(-90, 5)
ax_dq.set_title(r"Espectro de $d'_Q(t)$ (Após FPB)")
ax_dq.set_xlabel("Frequência (Hz)")
ax_dq.set_ylabel("Magnitude (dB)")
ax_dq.grid(True)
ax_dq.legend()
leg_dq = ax_dq.legend(
    loc='upper right', frameon=True, edgecolor='black',
    facecolor='white', fontsize=12, fancybox=True
)
leg_dq.get_frame().set_facecolor('white')
leg_dq.get_frame().set_edgecolor('black')
leg_dq.get_frame().set_alpha(1.0)

plt.tight_layout()
plt.subplots_adjust(top=0.92, hspace=0.4)
plt.savefig(os.path.join(output_dir, 'espectros_filtragem_comparacao.pdf'))





# === FILTRO CASADO ===
# Pulso casado: inverso no tempo do pulso rcc
g_matched = g[::-1]  # inverte o pulso

# Convolução dos sinais I e Q filtrados com o pulso casado
d_I_matched = np.convolve(d_I_rec, g_matched, mode='same')
d_Q_matched = np.convolve(d_Q_rec, g_matched, mode='same')

d_I_matched /= np.max(np.abs(d_I_matched))
d_Q_matched /= np.max(np.abs(d_Q_matched))
d_Q_matched *= -1

# Tempo ajustado
t_matched = np.arange(len(d_I_matched)) / fs

# === PLOT: Resposta ao impulso do filtro casado e sinais filtrados ===
fig_match = plt.figure(figsize=(16, 10))
gs_match = gridspec.GridSpec(3, 1)

# Resposta ao impulso do filtro casado
ax_mh = fig_match.add_subplot(gs_match[0])
ax_mh.plot(t_rc * 1000, g_matched, color='red', label='Resposta ao impulso - FC')
ax_mh.set_title("Resposta ao Impulso do Filtro Casado (RRC invertido)")
ax_mh.set_xlabel("Tempo (ms)")
ax_mh.set_ylabel("Amplitude")
ax_mh.grid(True)
ax_mh.legend()
leg_mh = ax_mh.legend(
    loc='upper right', frameon=True, edgecolor='black',
    facecolor='white', fontsize=12, fancybox=True
)
leg_mh.get_frame().set_facecolor('white')
leg_mh.get_frame().set_edgecolor('black')
leg_mh.get_frame().set_alpha(1.0)

# Canal I após filtro casado
ax_i_m = fig_match.add_subplot(gs_match[1])
ax_i_m.plot(t_matched, d_I_matched, color='blue', label='Canal I após Filtro Casado')
ax_i_m.set_title("Canal I(t) após Filtro Casado")
ax_i_m.set_xlim(0, 0.1)
ax_i_m.set_xlabel("Tempo (s)")
ax_i_m.set_ylabel("Amplitude")
ax_i_m.grid(True)
ax_i_m.legend()

leg_i_m = ax_i_m.legend(
    loc='upper right', frameon=True, edgecolor='black',
    facecolor='white', fontsize=12, fancybox=True
)
leg_i_m.get_frame().set_facecolor('white')
leg_i_m.get_frame().set_edgecolor('black')
leg_i_m.get_frame().set_alpha(1.0)

# Canal Q após filtro casado
ax_q_m = fig_match.add_subplot(gs_match[2])
ax_q_m.plot(t_matched, d_Q_matched, color='green', label='Canal Q após Filtro Casado')
ax_q_m.set_title("Canal Q(t) após Filtro Casado")
ax_q_m.set_xlim(0, 0.1)
ax_q_m.set_xlabel("Tempo (s)")
ax_q_m.set_ylabel("Amplitude")
ax_q_m.grid(True)
ax_q_m.legend()
leg_m = ax_q_m.legend(
    loc='upper right', frameon=True, edgecolor='black',
    facecolor='white', fontsize=12, fancybox=True
)
leg_m.get_frame().set_facecolor('white')
leg_m.get_frame().set_edgecolor('black')
leg_m.get_frame().set_alpha(1.0)

plt.tight_layout()
plt.subplots_adjust(top=0.92, hspace=0.4)

# Salvar figura
plt.savefig(os.path.join(output_dir, 'filtro_casado_resposta_e_sinais.pdf'))





# === ESPECTRO ANTES E APÓS FILTRAGEM CASADA ===
# FFT antes: sinais após passa-baixa
DI_f = np.fft.fftshift(np.fft.fft(d_I_rec))
DQ_f = np.fft.fftshift(np.fft.fft(d_Q_rec))

# FFT após: sinais após filtro casado
DIM_f = np.fft.fftshift(np.fft.fft(d_I_matched))
DQM_f = np.fft.fftshift(np.fft.fft(d_Q_matched))

freqs = np.fft.fftshift(np.fft.fftfreq(len(d_I_rec), d=1/fs))

DIM_db = mag2db(DIM_f)
DQM_db = mag2db(DQM_f)

# === PLOT: Espectros antes e depois da filtragem casada ===
fig_match_spec = plt.figure(figsize=(16, 10))
gs_match_spec = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])

# Linha 0: resposta ao impulso
ax_imp_m = fig_match_spec.add_subplot(gs_match_spec[0, :])
ax_imp_m.plot(t_rc * 1000, g_matched, color='red', linewidth=2, label='Resposta ao Impulso - FC')
ax_imp_m.set_title("Resposta ao Impulso do Filtro Casado (RRC invertido)")
ax_imp_m.set_xlabel("Tempo (ms)")
ax_imp_m.set_ylabel("Amplitude")
ax_imp_m.grid(True)
ax_imp_m.legend()

# Linha 1: Canal I — esquerda (ANTES), direita (APÓS)
ax_i_before = fig_match_spec.add_subplot(gs_match_spec[1, 0])
ax_i_before.plot(freqs, DI_db, color='navy', label=r"$|D_I(f)|$ antes FC")
ax_i_before.set_title(r"Espectro de $d'_I(t)$ (Antes de FC)")
ax_i_before.set_xlabel("Frequência (Hz)")
ax_i_before.set_ylabel("Magnitude (dB)")
ax_i_before.set_xlim(-fc, fc)
ax_i_before.set_ylim(-90, 5)
ax_i_before.grid(True)
ax_i_before.legend()

ax_i_after = fig_match_spec.add_subplot(gs_match_spec[1, 1])
ax_i_after.plot(freqs, DIM_db, color='navy', label=r"$|D_I(f)|$ após FC")
ax_i_after.set_title(r"Espectro de $d'_I(t)$ (Após FC)")
ax_i_after.set_xlabel("Frequência (Hz)")
ax_i_after.set_ylabel("Magnitude (dB)")
ax_i_after.set_ylim(-90, 5)
ax_i_after.set_xlim(-fc, fc)
ax_i_after.grid(True)
ax_i_after.legend()

# Linha 2: Canal Q — esquerda (ANTES), direita (APÓS)
ax_q_before = fig_match_spec.add_subplot(gs_match_spec[2, 0])
ax_q_before.plot(freqs, DQ_db, color='darkgreen', label=r"$|D_Q(f)|$ antes FC")
ax_q_before.set_title(r"Espectro de $d'_Q(t)$ (Antes de FC)")
ax_q_before.set_xlabel("Frequência (Hz)")
ax_q_before.set_ylabel("Magnitude (dB)")
ax_q_before.set_xlim(-fc, fc)
ax_q_before.set_ylim(-90, 5)
ax_q_before.grid(True)
ax_q_before.legend()

ax_q_after = fig_match_spec.add_subplot(gs_match_spec[2, 1])
ax_q_after.plot(freqs, DQM_db, color='green', label=r"$|D_Q(f)|$ após FC")
ax_q_after.set_title(r"Espectro de $d'_Q(t)$ (Após FC)")
ax_q_after.set_xlabel("Frequência (Hz)")
ax_q_after.set_ylabel("Magnitude (dB)")
ax_q_after.set_xlim(-fc, fc)
ax_q_after.set_ylim(-90, 5)
ax_q_after.grid(True)
ax_q_after.legend()

plt.tight_layout()
plt.subplots_adjust(top=0.93, hspace=0.4)

# Salvar
plt.savefig(os.path.join(output_dir, 'espectros_filtro_casado.pdf'))


# Atraso total causado pelos filtros: considera simetria do rcc e da convolução
delay_total = 0

# Índices ideais de amostragem
sample_indices = np.arange(delay_total, len(d_I_matched), sps)

# Protege contra índices fora dos limites do vetor
valid_indices = sample_indices[sample_indices < len(d_I_matched)]

# Sinais amostrados
I_samples = d_I_matched[valid_indices]
Q_samples = d_Q_matched[valid_indices]
t_samples = t_matched[valid_indices]

# === PLOT FINAL: Sinais IQ + pontos de amostragem ===
fig_sample = plt.figure(figsize=(16, 8))
gs_sample = gridspec.GridSpec(2, 1)

# Canal I
ax_si = fig_sample.add_subplot(gs_sample[0])
ax_si.plot(t_matched, d_I_matched, color='blue', label=r'Canal $I$ (Após FC)')
ax_si.stem(t_samples, I_samples, linefmt='k-', markerfmt='ko', basefmt=" ", label=r'Amostras $I$')
ax_si.set_title(r"Canal $I$ com Pontos de Decisão")
ax_si.set_xlabel("Tempo (s)")
ax_si.set_ylabel("Amplitude")
ax_si.set_xlim(0, 0.1)
ax_si.grid(True)
ax_si.legend()
leg_si = ax_si.legend(
    loc='upper right', frameon=True, edgecolor='black',
    facecolor='white', fontsize=12, fancybox=True
)
leg_si.get_frame().set_facecolor('white')
leg_si.get_frame().set_edgecolor('black')
leg_si.get_frame().set_alpha(1.0)

# Canal Q
ax_sq = fig_sample.add_subplot(gs_sample[1])
ax_sq.plot(t_matched, d_Q_matched, color='green', label=r'Canal $Q$ (Após FC)')
ax_sq.stem(t_samples, Q_samples, linefmt='k-', markerfmt='ko', basefmt=" ", label=r'Amostras $Q$')
ax_sq.set_title(r"Canal $Q$ com Pontos de Decisão")
ax_sq.set_xlabel("Tempo (s)")
ax_sq.set_ylabel("Amplitude")
ax_sq.set_xlim(0, 0.1)
ax_sq.grid(True)
ax_sq.legend()
leg_sq = ax_sq.legend(
    loc='upper right', frameon=True, edgecolor='black',
    facecolor='white', fontsize=12, fancybox=True
)
leg_sq.get_frame().set_facecolor('white')   
leg_sq.get_frame().set_edgecolor('black')
leg_sq.get_frame().set_alpha(1.0)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'sinais_amostrados_pos_filtro_casado.pdf'))



# === 21. Comparação entre bits amostrados e bits transmitidos ===

# Aplica correção de sinal ao Q
# Q_samples *= -1

# Limiarização para recuperação dos bits
I_rec_bits = (I_samples >= 0).astype(int)
Q_rec_bits = (Q_samples >= 0).astype(int)

# Bits originais
Xnrz_ref = Xnrz[:len(I_rec_bits)]
Ym_ref   = Ym[:len(Q_rec_bits)]

# Índices para eixo x
bit_idx = np.arange(len(I_rec_bits))

# === PLOT: Bits Originais vs Bits Recuperados ===
fig_bits = plt.figure(figsize=(16, 6))
gs_bits = gridspec.GridSpec(2, 1)

# Canal I (NRZ)
ax_i_bits = fig_bits.add_subplot(gs_bits[0])
ax_i_bits.step(bit_idx, Xnrz_ref, where='mid', label=r"NRZ original ($X_{NRZ}[n]$)", color='black', linestyle='--', linewidth=2)
ax_i_bits.step(bit_idx, I_rec_bits, where='mid', label=r"NRZ decodificado ($X'_{n}$)", color='blue')
ax_i_bits.set_title(r'Comparação NRZ - Original vs Decodificado')
ax_i_bits.set_ylabel(r'Canal $I$')
ax_i_bits.set_ylim(-0.2, 1.2)
ax_i_bits.grid(True)
ax_i_bits.legend(loc='upper right')
    
# Canal Q (Manchester)
ax_q_bits = fig_bits.add_subplot(gs_bits[1])
ax_q_bits.step(bit_idx, Ym_ref, where='mid', label=r"Manchester original ($Y_{M}[n]$)", color='black', linestyle='--', linewidth=2)
ax_q_bits.step(bit_idx, Q_rec_bits, where='mid', label=r"Manchester decodificado ($Y'_{n}$)", color='red')
ax_q_bits.set_title(r'Comparação Manchester - Original vs Decodificado')
ax_q_bits.set_xlabel(r'Índice de bit')
ax_q_bits.set_ylabel(r'Canal $Q$')
ax_q_bits.set_ylim(-0.2, 1.2)
ax_q_bits.grid(True)
ax_q_bits.legend(loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'comparacao_bits_recuperados.pdf'))
