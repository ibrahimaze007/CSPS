import os
from ucimlrepo import fetch_ucirepo

concrete = fetch_ucirepo(id=165)
X = concrete.data.features.copy()
y = concrete.data.targets.iloc[:, 0].rename("strength")

print("=== Dataset Metadata ===")
print(concrete.metadata)
print("\n=== Variables Table ===")
print(concrete.variables)

X.columns = [
    "cement", "slag", "ash", "water",
    "superplastic", "coarse_agg", "fine_agg", "age"
]

os.makedirs("data", exist_ok=True)
X.assign(strength=y).to_csv("data/concrete.csv", index=False)
print("\nSaved data/concrete.csv successfully.")
