import numpy as np
import matplotlib.pyplot as plt
import os

# 설정
base_dir = "/jsm0707/GENIE/train_output/PACS"
algorithms = [
    "250327_13-49-52_adam_sharpness"
]
test_env = "[0]"
rho_index_to_plot = 2  # 예: ρ = 0.03

plt.figure(figsize=(8, 5))

for algo in algorithms:
    csv_path = os.path.join(
        base_dir, "ERM", test_env, algo, "sharpness_progress", "sharpness_over_time.csv"
    )

    if not os.path.exists(csv_path):
        print(f"❌ CSV not found for {algo}: {csv_path}")
        continue

    data = np.loadtxt(csv_path, delimiter=",")
    steps = data[:, 0]
    sharpness = data[:, 1 + rho_index_to_plot]  # 선택한 rho에 해당하는 열

    plt.plot(steps, sharpness, label=f"{algo} (ρ = {0.01*(rho_index_to_plot+1):.2f})")

plt.xlabel("Training Step")
plt.ylabel(r"$h_\rho(\theta)$")
plt.title("Sharpness Over Training")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("sharpness_comparison.png")
plt.show()