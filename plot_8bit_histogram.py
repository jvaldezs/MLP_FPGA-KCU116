import joblib
import numpy as np
import os
import matplotlib.pyplot as plt

# Paths
BASE_PKL = r"c:\Users\jvald\HLS_Export\Weights\TPU_CODE\base.pkl"
WEIGHTS_PKL = r"c:\Users\jvald\HLS_Export\Weights\TPU_CODE\weights.pkl"
OUTPUT_IMG = r"C:\Users\jvald\.gemini\antigravity\brain\d7ae2e1b-3951-4b14-9168-71e761676b1f\quantization_8bit_report.png"

def generate_report():
    if not os.path.exists(BASE_PKL) or not os.path.exists(WEIGHTS_PKL):
        print("Error: Pickle files not found.")
        return

    print("Loading weights...")
    W1 = np.array(joblib.load(BASE_PKL)).flatten()
    W2 = np.array(joblib.load(WEIGHTS_PKL)).flatten()

    # ap_fixed<8, 4> parameters:
    # 4 integer bits (including sign)
    # 4 fractional bits
    # Resolution = 2^-4 = 0.0625
    # Rounding threshold = 2^-5 = 0.03125
    threshold = 2.0 ** -5
    
    print(f"\nAnalysis for ap_fixed<8, 4> (4 fractional bits)")
    print(f"Rounding threshold (values smaller than this become 0): {threshold:.10f}")

    def get_stats(data, name):
        total = len(data)
        original_zeros = np.sum(data == 0)
        # We only care about numbers that ARE NOT already zero but round to zero
        rounded_to_zero = np.sum((np.abs(data) < threshold) & (data != 0))
        pct = (rounded_to_zero / total) * 100
        return original_zeros, rounded_to_zero, pct

    w1_orig, w1_round, w1_pct = get_stats(W1, "W1")
    w2_orig, w2_round, w2_pct = get_stats(W2, "W2")

    print("\n--- Histogram Report ---")
    print(f"{'Layer':<10} | {'Total Weights':<15} | {'Already Zero':<15} | {'Newly Rounded':<15} | {'Percentage Loss':<15}")
    print("-" * 80)
    print(f"{'W1':<10} | {len(W1):<15,} | {w1_orig:<15,} | {w1_round:<15,} | {w1_pct:.6f}%")
    print(f"{'W2':<10} | {len(W2):<15,} | {w2_orig:<15,} | {w2_round:<15,} | {w2_pct:.6f}%")

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    def plot_layer(ax, data, name, rounded_pct, threshold):
        # Focus on the distribution around zero
        zoom_range = 0.5 # Larger zoom for 8-bit
        data_zoom = data[(data > -zoom_range) & (data < zoom_range)]
        
        ax.hist(data_zoom, bins=100, color='lightcoral', edgecolor='black', alpha=0.7)
        ax.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold (+{threshold:.5f})')
        ax.axvline(x=-threshold, color='red', linestyle='--')
        ax.fill_betweenx([0, ax.get_ylim()[1]], -threshold, threshold, color='red', alpha=0.2, label='Zero Zone (Rounded to 0)')
        
        ax.set_title(f"Weight Distribution: {name}\n({rounded_pct:.4f}% Newly Rounded to Zero)")
        ax.set_xlabel("Weight Value")
        ax.set_ylabel("Frequency")
        ax.legend()

    plot_layer(ax1, W1, "W1 (Hidden)", w1_pct, threshold)
    plot_layer(ax2, W2, "W2 (Output)", w2_pct, threshold)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG)
    print(f"\nHistogram saved to: {OUTPUT_IMG}")

if __name__ == "__main__":
    generate_report()
