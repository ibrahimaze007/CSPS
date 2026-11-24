import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

d = pd.read_csv("data/seismogram_A.csv")
t = d["time_s"].to_numpy()
Z, N, E = d["Z"].to_numpy(), d["N"].to_numpy(), d["E"].to_numpy()
fs = 1.0 / (t[1] - t[0])
print(f"Sampling rate fs = {fs:.1f} Hz, samples = {len(t)}, duration = {t[-1]:.1f} s")

# Step 1: Three channels in mm/s + zoom on first 60s
fig, axes = plt.subplots(3, 2, figsize=(12, 7))
for i, (ch, name) in enumerate(zip([Z, N, E], ["Z (Vertical)", "N (North-South)", "E (East-West)"])):
    axes[i, 0].plot(t, ch * 1e3, lw=0.6)
    axes[i, 0].set_ylabel(f"{name}\n(mm/s)")
    axes[i, 0].grid(True, alpha=0.3)
    mask60 = t <= 60
    axes[i, 1].plot(t[mask60], ch[mask60] * 1e3, lw=0.6, color="tab:orange")
    axes[i, 1].grid(True, alpha=0.3)
axes[0, 0].set_title("Full Record (mm/s)")
axes[0, 1].set_title("First 60 s Zoom (Background Noise)")
axes[2, 0].set_xlabel("Time (s)")
axes[2, 1].set_xlabel("Time (s)")
plt.tight_layout()
plt.savefig("results/fig1_records.png")
plt.close()

# Step 2: FFT Spectra
N_pts = len(Z)
X = np.fft.fft(Z)
freqs = np.fft.fftfreq(N_pts, d=1/fs)
amp = np.abs(X) / N_pts * 2

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].plot(np.arange(N_pts), amp, lw=0.6)
axes[0].set_title("2(a) Amplitude vs Array Index")
axes[0].set_xlabel("Index")
axes[0].set_ylabel("Amplitude")

axes[1].plot(freqs, amp, lw=0.6, color="tab:green")
axes[1].set_title("2(a) Amplitude vs Frequency (Hz)")
axes[1].set_xlabel("Frequency (Hz)")

pos = freqs > 0
Nq = int(60 * fs)
fq = np.fft.fftfreq(Nq, d=1/fs)
amp_q = np.abs(np.fft.fft(Z[:Nq])) / Nq * 2
pos_q = fq > 0

axes[2].loglog(freqs[pos], amp[pos], lw=0.6, label="Whole Record", alpha=0.8)
axes[2].loglog(fq[pos_q], amp_q[pos_q], lw=0.7, label="First 60 s (Quiet)", alpha=0.85)
axes[2].set_title("2(b) Log-Log Spectrum (Positive Freqs)")
axes[2].set_xlabel("Frequency (Hz)")
axes[2].legend()
for ax in axes:
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/fig2_spectra.png")
plt.close()

# Step 3: Inverse FFT & Band-pass filter (0.5 - 10 Hz)
assert np.allclose(np.fft.ifft(np.fft.fft(Z)).real, Z)
print("Step 3(a) np.allclose(ifft(fft(Z)).real, Z) -> True")

def bandpass(x, fs, a=0.5, b=10.0):
    X_f = np.fft.fft(x)
    f_arr = np.fft.fftfreq(len(x), d=1/fs)
    X_f[(np.abs(f_arr) < a) | (np.abs(f_arr) > b)] = 0
    return np.fft.ifft(X_f).real

Z_clean = bandpass(Z, fs, 0.5, 10.0)
plt.figure(figsize=(10, 3.5))
plt.plot(t, (Z - Z_clean) * 1e3, lw=0.7, color="crimson")
plt.title("Step 3(c) Removed Signal: Z - clean (mm/s)")
plt.xlabel("Time (s)")
plt.ylabel("Removed (mm/s)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/fig3_removed_noise.png")
plt.close()

# Step 4: Envelopes with 0.05s, 0.5s, and 5s windows
def envelope(x, win_s, fs):
    w = max(1, int(round(win_s * fs)))
    return np.convolve(np.abs(x), np.ones(w) / w, mode="same")

plt.figure(figsize=(10, 4))
for ws, col in zip([0.05, 0.5, 5.0], ["lightgray", "tab:blue", "tab:red"]):
    plt.plot(t, envelope(Z_clean, ws, fs) * 1e3, label=f"Window = {ws} s", color=col, lw=1.1 if ws!=0.05 else 0.6)
plt.title("Step 4: Effect of Smoothing Window on Filtered Z Envelope")
plt.xlabel("Time (s)")
plt.ylabel("Envelope (mm/s)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/fig4_envelopes.png")
plt.close()
print("Saved fig1 through fig4 successfully.")
