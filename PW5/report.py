import pandas as pd

df = pd.read_csv("results/events.csv")
lines = ["=== Earthquake Seismic Analysis Report ===\n"]
for _, r in df.iterrows():
    lines.append(
        f"Event {r['event']}: P-arrival = {r['t_P_s']} s, S-arrival = {r['t_S_s']} s, "
        f"Distance R = {r['R_km']} km, PGV = {r['PGV_cm_s']} cm/s, Magnitude M = {r['Magnitude']}\n"
    )

closer = df.loc[df["R_km"].idxmin(), "event"]
stronger = df.loc[df["Magnitude"].idxmax(), "event"]
lines.append(f"\nComparison: {closer} was closer to the station, while {stronger} was stronger (higher magnitude).\n")

with open("results/report.txt", "w") as f:
    f.writelines(lines)
print("".join(lines))
