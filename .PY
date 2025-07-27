import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Parámetros básicos
Fs = 10000  # Frecuencia de muestreo
t = np.arange(0, 1, 1/Fs)  # Tiempo: 1 segundo

# Señal de mensaje (baja frecuencia)
fm = 50  # Hz
Am = 1
mensaje = Am * np.sin(2 * np.pi * fm * t)

# Señal portadora (alta frecuencia)
fc = 1000  # Hz
Ac = 2
portadora = Ac * np.cos(2 * np.pi * fc * t)

# Señal modulada en amplitud
modulada = (1 + mensaje) * portadora

# Señal con ruido
ruido = np.random.normal(0, 0.5, len(t))
modulada_ruido = modulada + ruido

# Función para graficar en dominio de frecuencia
def graficar_fft(signal, Fs, title):
    N = len(signal)
    fft_result = np.abs(fft(signal)) / N
    freqs = fftfreq(N, 1/Fs)
    idx = np.where(freqs >= 0)

    plt.plot(freqs[idx], fft_result[idx])
    plt.title(title)
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Magnitud")
    plt.grid()

# --------- GRAFICACIÓN ---------

plt.figure(figsize=(15, 10))

# Señal original (mensaje)
plt.subplot(3, 2, 1)
plt.plot(t[:1000], mensaje[:1000])
plt.title("Señal de mensaje (baja frecuencia)")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")

# Señal portadora
plt.subplot(3, 2, 2)
plt.plot(t[:1000], portadora[:1000])
plt.title("Portadora (alta frecuencia)")
plt.xlabel("Tiempo (s)")

# Señal modulada AM
plt.subplot(3, 2, 3)
plt.plot(t[:1000], modulada[:1000])
plt.title("Señal Modulada AM (Tiempo)")
plt.xlabel("Tiempo (s)")

# Espectro frecuencia de señal modulada
plt.subplot(3, 2, 4)
graficar_fft(modulada, Fs, "FFT de señal modulada")

# Señal con ruido
plt.subplot(3, 2, 5)
plt.plot(t[:1000], modulada_ruido[:1000])
plt.title("Señal Modulada con Ruido")
plt.xlabel("Tiempo (s)")

# Espectro frecuencia con ruido
plt.subplot(3, 2, 6)
graficar_fft(modulada_ruido, Fs, "FFT de señal con ruido")

plt.tight_layout()
plt.show()
