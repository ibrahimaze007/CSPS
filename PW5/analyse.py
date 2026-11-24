import os
import numpy as np
import pandas as pd

def bandpass(x, fs, a=0.5, b=10.0):
    X_f = np.fft.fft(x)
    f_arr = np.fft.fftfreq(len(x), d=1/fs)
    X_f[(np.abs(f_arr) < a) | (np.abs(f_arr) > b)] = 0
    return np.fft.ifft(X_f).real

def envelope(x, win_s, fs):
    w = max(1, int(round(win_s * fs)))
    return np.convolve(np.abs(x), np.ones(w) / w, mode="same")

def analyse(csv_file):
    d = pd.read_csv(csv_file)
    t = d["time_s"].to_numpy()
    Z, N, E = d["Z"].to_numpy(), d["N"].to_numpy(), d["E"].to_numpy()
    fs = 1.0 / (t[1] - t[0])

    Zc, Nc, Ec = bandpass(Z, fs), bandpass(N, fs), bandpass(E, fs)
    env_Z = envelope(Zc, 0.5, fs)
    env_H = envelope(np.sqrt(Nc**2 + Ec**2), 0.5, fs)

    t_P = t[np.argmax(env_Z > 0.05 * np.max(env_Z))]
    after_P = t > (t_P + 2.0)
    t_S = t[np.argmax(after_P & (env_H > 0.10 * np.max(env_H)))]

    v_P, v_S = 6.2, 3.6
    R = (t_S - t_P) / (1.0 / v_S - 1.0 / v_P)
    pgv_cms = np.max(np.sqrt(N**2 + E**2)) * 100.0
    M = (np.log10(pgv_cms) + np.log10(R) + 1.0) / 0.55

    return {
        "event": os.path.basename(csv_file),
        "t_P_s": round(float(t_P), 2),
        "t_S_s": round(float(t_S), 2),
        "R_km": round(float(R), 2),
        "PGV_cm_s": round(float(pgv_cms), 4),
        "Magnitude": round(float(M), 2)
    }

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    rows = [analyse("data/seismogram_A.csv"), analyse("data/seismogram_B.csv")]
    df_res = pd.DataFrame(rows)
    df_res.to_csv("results/events.csv", index=False)
    print(df_res.to_string(index=False))
