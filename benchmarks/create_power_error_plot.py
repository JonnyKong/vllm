import json
import matplotlib.pyplot as plt
import numpy as np

with open("power_error_data.json", "r") as f:
    power_errors = json.load(f)

for label in power_errors.keys():
    print(label)

gpu_to_key = {
    "T4" : "T4 phi-2",
    "A40" : "A40 Llama-3.1-8B-Instruct",
    "A100" : "A100-SXM4-80GB gemma-2-27b-it",
    "H100" : "H100-80GB-HBM3 gemma-2-27b-it",
    "A100-TP4" : "A100-SXM4-80GB Llama-3.1-70B-Instruct"
}

gpu_to_style = {
    'T4' : '-',
    'A40' : '--',
    'A100' : '-.',
    'H100' : '-.',
    'A100-TP4' : ':'
}
gpu_to_color = {
    'T4' : '#1f77b4',
    'A40' : '#ff7f0e',
    'A100' : '#2ca02c',
    'H100' : '#d62728',
    'A100-TP4' : '#9467bd'
}

fig, ax = plt.subplots(figsize=(4, 2.5))

for label, color in gpu_to_color.items():
    sorted_diff = np.sort(power_errors[gpu_to_key[label]])
    cdf = np.arange(len(sorted_diff)) / len(sorted_diff) * 100  # Convert to percent
    ax.plot(sorted_diff, cdf, label=label, linewidth=2, color=color, linestyle=gpu_to_style[label])

ax.set_xlabel('Absolute power prediction error (%)')
ax.set_ylabel('CDF')
ax.tick_params(axis='both')
ax.legend()
ax.grid(False)
plt.tight_layout()
plt.savefig('absolute_power_error_cdf.pdf', bbox_inches="tight")
plt.close()