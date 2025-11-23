# %%
from IPython.display import HTML
from matplotlib import pyplot as plt, animation
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scienceplots
import os
import numpy as np


plt.style.use('science')
plt.rcParams["figure.figsize"] = (16, 11)

plt.rc('font', size=16)          # tamanho da fonte geral (eixos, ticks)
plt.rc('axes', titlesize=22)     # tamanho da fonte do título dos eixos
plt.rc('axes', labelsize=22)     # tamanho da fonte dos rótulos dos eixos (xlabel, ylabel)
plt.rc('xtick', labelsize=16)    # tamanho da fonte dos ticks no eixo x
plt.rc('ytick', labelsize=16)    # tamanho da fonte dos ticks no eixo y
plt.rc('legend', fontsize=16)    # tamanho da fonte das legendas (se houver)
plt.rc('figure', titlesize=22)   # tamanho da fonte do título da figura (plt.suptitle)




# %%
def gerar_sinal_simulado():
    fs = 20e6
    duracao = 1.2  # 600 ms
    t = np.arange(0, duracao, 1/fs)

    sinal = np.random.normal(0, 0.5, len(t))  # ruído de fundo

    # Criar várias portadoras de 40 ms
    num_portadoras = 3
    dur_portadora = 0.08  # 40 ms
    N_portadora = int(fs * dur_portadora)

    portadoras_info = []

    for _ in range(num_portadoras):
        t_inicio = np.random.uniform(0.05, duracao - dur_portadora - 0.05)
        f_carry = np.random.uniform(401.62e3, 401.65e3)  # em Hz

        idx_inicio = int(t_inicio * fs)
        idx_fim = idx_inicio + N_portadora
        portadora = 5 * np.cos(2 * np.pi * f_carry * t[:N_portadora])

        sinal[idx_inicio:idx_fim] += portadora
        portadoras_info.append((t_inicio, f_carry / 1e3))  # salvar info em kHz

    return t, sinal, fs, portadoras_info


# %%
def adicionar_awgn(sinal, snr_dB):
    """
    Adiciona ruído AWGN ao sinal com um determinado SNR (Signal-to-Noise Ratio).
    
    :param sinal: O sinal original
    :param snr_dB: O SNR em dB
    :return: O sinal com ruído AWGN adicionado
    """
    # Calcular a potência do sinal original
    sinal_power = np.mean(sinal**2)
    
    # Calcular a potência do ruído desejado
    snr_linear = 10**(snr_dB / 10)
    ruido_power = sinal_power / snr_linear
    
    # Gerar o ruído AWGN
    ruido = np.random.normal(0, np.sqrt(ruido_power), len(sinal))
    
    # Adicionar o ruído ao sinal original
    sinal_com_ruido = sinal + ruido
    return sinal_com_ruido


# %%
def detectar_portadora(sinal, fs):
    janela_ms = 10
    N_janela = int(fs * (janela_ms / 1000))
    num_quadros = len(sinal) // N_janela

    resultados = []

    for i in range(num_quadros):
        ini = i * N_janela
        fim = ini + N_janela
        janela = sinal[ini:fim]

        # FFT normalizada
        fft = np.fft.fft(janela)
        freqs = np.fft.fftfreq(len(janela), 1/fs)
        idx_pos = freqs >= 0
        freqs = freqs[idx_pos] / 1e3  # para kHz
        fft = (2 / len(janela)) * np.abs(fft[idx_pos])

        # Verifica a frequência mais próxima de 401.63 kHz com janela ±200 Hz
        f_alvo = 401.635  # centro da faixa
        delta_f = 0.2     # kHz (±200 Hz)
        
        faixa_idx = np.where((freqs >= f_alvo - delta_f) & (freqs <= f_alvo + delta_f))[0]
        
        if len(faixa_idx) > 0:
            max_amp = np.max(fft[faixa_idx])
            detectado = max_amp >= 2.0
        else:
            detectado = False

        detectado = max_amp >= 2.0

        resultados.append(detectado)

    # Aplica critério: 2 quadros consecutivos com detecção
    confirmados = [False] * len(resultados)
    for i in range(len(resultados) - 1):
        if resultados[i] and resultados[i+1]:
            confirmados[i] = True
            confirmados[i+1] = True

    return resultados, confirmados

# %%
def salvar_quadro_pdf(args):
    i, t, sinal, fs, resultados, confirmados, output_dir, janela_ms, snr_dB, limiar_fft, faixa_freq, figsize = args
    # Janela atual do sinal
    N_janela = int(fs * (janela_ms / 1000))
    ini = i * N_janela
    fim = ini + N_janela
    janela = sinal[ini:fim]
    t_janela = (t[ini:fim] - t[ini]) * 1000  # tempo em ms
    # Adiciona ruído AWGN
    janela_com_ruido = adicionar_awgn(janela, snr_dB=snr_dB)
    # FFT com zero-padding para interpolação
    N_fft = 4 * N_janela
    fft_complexo = np.fft.fft(janela_com_ruido, n=N_fft)
    freqs = np.fft.fftfreq(N_fft, 1/fs)
    idx_pos = freqs >= 0
    freqs = freqs[idx_pos] / 1e3
    fft = np.abs(fft_complexo[idx_pos])
    fft *= np.max(np.abs(janela_com_ruido)) / np.max(fft)
    margem_amp = 0.2
    y_max_amp = np.max(np.abs(janela_com_ruido)) * (1 + margem_amp)
    margem_fft = 0.2
    y_max_fft = np.max(fft) * (1 + margem_fft)
    fig, axs = plt.subplots(2, 1, figsize=figsize)
    axs[0].plot(t_janela, janela_com_ruido, lw=2, color='k')
    axs[0].set_xlim(0, janela_ms)
    axs[0].set_ylim(-y_max_amp, y_max_amp)
    axs[0].set_ylabel('Amplitude (V)')
    axs[0].set_xlabel('Tempo (ms)', labelpad=10)
    axs[0].set_title('Sinal no Tempo')
    axs[0].grid(True, linestyle='--', alpha=0.7)
    axs[1].plot(freqs, fft, lw=2, color='k', label='FFT')
    axs[1].axhline(y=limiar_fft, color='r', linestyle='--', label=f'Threshold ({limiar_fft}V)')
    axs[1].set_xlim(faixa_freq)
    axs[1].set_ylim(0, max(y_max_fft, limiar_fft * 1.1))
    axs[1].set_xlabel('Frequência (kHz)', labelpad=10)
    axs[1].set_ylabel('Magnitude (V)')
    axs[1].set_title('FFT da Janela')
    axs[1].legend(
        loc='upper right',
        frameon=True,
        edgecolor='black',
        facecolor='white',
        fontsize=12,
        fancybox=True
    )
    axs[1].grid(True, linestyle='--', alpha=0.7)
    status = 'COM PORTADORA' if confirmados[i] else 'SEM PORTADORA'
    cor_status = 'black'
    fig.suptitle(f'Tempo: {i*janela_ms} ms — {status}', fontsize=16, color=cor_status)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    nome_arquivo = os.path.join(output_dir, f'quadro{i+1}.pdf')
    plt.savefig(nome_arquivo, format='pdf')
    plt.close(fig)
    return nome_arquivo

def animar_detecção_em_imagens(t, sinal, fs, resultados, confirmados, output_dir=None, n_workers=None):
    from tqdm import tqdm
    import os
    # CONFIGURAÇÕES (alteráveis facilmente)
    janela_ms = 10
    snr_dB = 10
    limiar_fft = 2.0
    faixa_freq = (400, 403)
    figsize = (16, 9)
    salvar_debug = True
    if output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, '..', '..', 'out', 'detection')
    os.makedirs(output_dir, exist_ok=True)
    N_janela = int(fs * (janela_ms / 1000))
    num_quadros = len(sinal) // N_janela
    args_list = [
        (i, t, sinal, fs, resultados, confirmados, output_dir, janela_ms, snr_dB, limiar_fft, faixa_freq, figsize)
        for i in range(num_quadros)
    ]
    if n_workers is None:
        n_workers = os.cpu_count()
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(salvar_quadro_pdf, args) for args in args_list]
        for f in tqdm(as_completed(futures), total=len(futures), desc='Exportando PDFs'):
            nome_arquivo = f.result()
            if salvar_debug:
                print(f'Salvo: {nome_arquivo}')

# %%
t, sinal, fs, portadoras_info = gerar_sinal_simulado()
resultados, confirmados = detectar_portadora(sinal, fs)
animar_detecção_em_imagens(t, sinal, fs, resultados, confirmados)



