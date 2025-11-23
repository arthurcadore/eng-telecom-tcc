import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, cheby1, ellip, bessel, freqz

# Parâmetros
ordem = 6
fc = 1500  # Frequência de corte (Hz)
fs = 20000  # Frequência de amostragem (Hz)
w_c = 2 * fc / fs  # Frequência normalizada

# Projetar filtros
b_butter, a_butter = butter(ordem, w_c)
b_cheby, a_cheby = cheby1(ordem, 1, w_c)       # ripple = 1 dB
b_ellip, a_ellip = ellip(ordem, 1, 40, w_c)    # ripple = 1 dB, attenuação = 40 dB
b_bessel, a_bessel = bessel(ordem, w_c, norm='phase')

# Calcular resposta em frequência
w, h_butter = freqz(b_butter, a_butter, worN=1024, fs=fs)
_, h_cheby  = freqz(b_cheby,  a_cheby,  worN=1024, fs=fs)
_, h_ellip  = freqz(b_ellip,  a_ellip,  worN=1024, fs=fs)
_, h_bessel = freqz(b_bessel, a_bessel, worN=1024, fs=fs)

# Limite inferior de dB para evitar log de zero
def to_dB(h):
    mag = 20 * np.log10(np.abs(h))
    mag = np.clip(mag, -100, None)  # Limita em -100 dB
    return mag

# Plot
plt.figure(figsize=(16, 9))

plt.subplot(2, 2, 1)
plt.plot(w, to_dB(h_butter))
plt.title('Butterworth')
plt.xlabel('Frequência (Hz)')
plt.ylabel('Magnitude (dB)')
plt.ylim([-100, 5])
plt.xlim(0, fs / 4)
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(w, to_dB(h_cheby))
plt.title('Chebyshev Tipo I')
plt.xlabel('Frequência (Hz)')
plt.ylabel('Magnitude (dB)')
plt.ylim([-100, 5])
plt.xlim(0, fs / 4)
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(w, to_dB(h_ellip))
plt.title('Elíptico')
plt.xlabel('Frequência (Hz)')
plt.ylabel('Magnitude (dB)')
plt.ylim([-100, 5])
plt.xlim(0, fs / 4)
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(w, to_dB(h_bessel))
plt.title('Bessel')
plt.xlabel('Frequência (Hz)')
plt.ylabel('Magnitude (dB)')
plt.ylim([-100, 5])
plt.xlim(0, fs / 4)
plt.grid(True)

plt.tight_layout()
plt.savefig('../../out/filters_response.pdf', dpi=300, bbox_inches='tight')
plt.show()


# Plot
plt.figure(figsize=(16, 9))

for ordem in range(2, 7):  # Ordens de 1 a 6
    if ordem == 0:
        # Ordem 0: ganho constante (filtro passa-baixa idealmente plano)
        w = np.linspace(0, fs / 2, 1024)
        h = np.ones_like(w)
    else:
        b, a = butter(ordem, w_c)
        w, h = freqz(b, a, worN=1024, fs=fs)
    plt.plot(w, to_dB(h), label=f'Ordem {ordem}')

plt.title('Resposta em frequência - Filtro Butterworth')
plt.xlabel('Frequência (Hz)')
plt.ylabel('Magnitude (dB)')
plt.ylim([-100, 5])
plt.xlim(0, fs / 4)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('../../out/butterworth_0a6.pdf', dpi=300, bbox_inches='tight')
plt.show()
