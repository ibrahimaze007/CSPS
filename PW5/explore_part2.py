import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, spectrogram

d = pd.read_csv("data/seismogram_A.csv")
t = d["time_s"].to_numpy()
Z, N, E = d["Z"].to_numpy(), d["N"].to_numpy(), d["E"].to_numpy()
fs = 1.0 / (t[1] - t[0])

def bandpass(x, fs, a=0.5, b=10.0):
    X_f = np.fft.fft(x)
    f_arr = np.fft.fftfreq(len(x), d=1/fs)
    X_f[(np.abs(f_arr) < a) | (np.abs(f_arr) > b)] = 0
    return np.fft.ifft(X_f).real

def envelope(x, win_s, fs):
    w = max(1, int(round(win_s * fs)))
    return np.convolve(np.abs(x), np.ones(w) / w, mode="same")

Zc, Nc, Ec = bandpass(Z, fs), bandpass(N, fs), bandpass(E, fs)
env_Z = envelope(Zc, 0.5, fs)
env_H = envelope(np.sqrt(Nc**2 + Ec**2), 0.5, fs)

# Step 5: P and S arrival times
idx_P = np.argmax(env_Z > 0.05 * np.max(env_Z))
t_P = t[idx_P]
after_P = t > (t_P + 2.0)
idx_S = np.argmax(after_P & (env_H > 0.10 * np.max(env_H)))
t_S = t[idx_S]

peaks, _ = find_peaks(env_H, height=0.5 * np.max(env_H))
t_max_H = t[peaks[np.argmax(env_H[peaks])]]

# Step 6: Distance & Origin time
v_P, v_S = 6.2, 3.6
R = (t_S - t_P) / (1.0 / v_S - 1.0 / v_P)
dR_1s = 1.0 / (1.0 / v_S - 1.0 / v_P)
t_0 = t_P - R / v_P

# Step 8: Magnitude
pgv_cms = np.max(np.sqrt(N**2 + E**2)) * 100.0
M = (np.log10(pgv_cms) + np.log10(R) + 1.0) / 0.55

print(f"Dataset A -> t_P = {t_P:.2f} s, t_S = {t_S:.2f} s (max H peak at {t_max_H:.2f} s)")
print(f"Distance R = {R:.2f} km (±{dR_1s:.2f} km per 1s t_S error), Origin time t_0 = {t_0:.2f} s")
print(f"PGV = {pgv_cms:.4f} cm/s, Magnitude M = {M:.2f}")

# Plot Step 5 arrivals on 3 channels
fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
for ax, ch, name in zip(axes, [Zc, Nc, Ec], ["Filtered Z", "Filtered N", "Filtered E"]):
    ax.plot(t, ch * 1e3, lw=0.6)
    ax.axvline(t_P, color="green", ls="--", lw=1.4, label=f"t_P = {t_P:.2f} s")
    ax.axvline(t_S, color="red", ls="--", lw=1.4, label=f"t_S = {t_S:.2f} s")
    ax.axvline(t_max_H, color="purple", ls=":", lw=1.2, label=f"Max H = {t_max_H:.2f} s")
    ax.set_ylabel(f"{name}\n(mm/s)")
    ax.grid(True, alpha=0.3)
axes[0].legend(loc="upper right")
axes[0].set_title("Step 5: P-wave and S-wave Arrivals Marked on Filtered Channels")
axes[2].set_xlabel("Time (s)")
plt.tight_layout()
plt.savefig("results/fig5_arrivals.png")
plt.close()

# Step 7: Spectrogram of raw Z up to 10 Hz
f_sp, ts_sp, S_sp = spectrogram(Z, fs=fs, nperseg=256)
mask_10 = f_sp <= 10.0
plt.figure(figsize=(10, 4.5))
plt.pcolormesh(ts_sp, f_sp[mask_10], 10 * np.log10(S_sp[mask_10] + 1e-30), shading="gouraud", cmap="inferno")
plt.axvline(t_P, color="cyan", ls="--", label="P arrival")
plt.axvline(t_S, color="lime", ls="--", label="S arrival")
plt.colorbar(label="Power Spectral Density (dB)")
plt.ylabel("Frequency (Hz)")
plt.xlabel("Time (s)")
plt.title("Step 7: Spectrogram of Raw Z Channel (0–10 Hz)")
plt.legend()
plt.tight_layout()
plt.savefig("results/fig7_spectrogram.png")
plt.close()
print("Saved fig5 and fig7 successfully.")
