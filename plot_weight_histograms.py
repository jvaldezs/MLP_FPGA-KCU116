import joblib
import numpy as np
import matplotlib.pyplot as plt
import os

# Paths
DIR = r"c:\Users\jvald\HLS_Export\Weights\TPU_CODE"
FILES = {
    "W1 (base)": "base.pkl",
    "b1 (bias)": "bias.pkl",
    "W2 (weights)": "weights.pkl",
    "b2 (bias2)": "bias2.pkl"
}
ARTIFACTS_DIR = r"C:\Users\jvald\.gemini\antigravity\brain\6ca50188-560e-4388-be77-63685cb8eff7"

def plot_histograms():
    plt.figure(figsize=(15, 10))
    
    for i, (name, filename) in enumerate(FILES.items(), 1):
        path = os.path.join(DIR, filename)
        if not os.path.exists(path):
            print(f"Skipping {name} - not found")
            continue
            
        data = joblib.load(path)
        weights = np.array(data).flatten()
        
        plt.subplot(2, 2, i)
        plt.hist(weights, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
        plt.title(f"Weight Distribution: {name}")
        plt.xlabel("Weight Value")
        plt.ylabel("Frequency")
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Add stats
        mean_val = np.mean(weights)
        std_val = np.std(weights)
        plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_val:.4f}')
        
        # Calculate percentage near zero for 4-bit (Threshold 1.0)
        threshold_4bit = 1.0
        near_zero_4bit = np.sum(np.abs(weights) < (threshold_4bit / 2)) / len(weights) * 100
        
        # Calculate percentage near zero for 16-bit (ap_fixed<16, 4>)
        # LSB = 2^-12 = 0.000244140625. Round to zero if < half LSB.
        threshold_16bit = 2**-12
        near_zero_16bit = np.sum(np.abs(weights) < (threshold_16bit / 2)) / len(weights) * 100
        
        # Calculate percentage near zero for 26-bit (ap_fixed<26, 4>)
        # LSB = 2^-22 = 0.000000238418579. Round to zero if < half LSB.
        threshold_26bit = 2**-22
        near_zero_26bit = np.sum(np.abs(weights) < (threshold_26bit / 2)) / len(weights) * 100
        
        # Calculate percentage of EXACT zeros
        exact_zeros = np.sum(weights == 0.0) / len(weights) * 100
        
        plt.annotate(f'Exact Zeros: {exact_zeros:.2f}%\n'
                    f'Round to 0 (4-bit): {near_zero_4bit:.2f}%\n'
                    f'Round to 0 (16-bit): {near_zero_16bit:.2f}%\n'
                    f'Round to 0 (26-bit): {near_zero_26bit:.2f}%', 
                    xy=(0.05, 0.70), xycoords='axes fraction', 
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=1))
        
        print(f"{name}:")
        print(f"  Total weights: {len(weights)}")
        print(f"  Exact zeros: {exact_zeros:.4f}%")
        print(f"  Rounded to 0 @ 4-bit  (I=4): {near_zero_4bit:.4f}%")
        print(f"  Rounded to 0 @ 16-bit (I=4, F=12): {near_zero_16bit:.4f}%")
        print(f"  Rounded to 0 @ 26-bit (I=4, F=22): {near_zero_26bit:.4f}%")
        
        plt.legend()

    plt.tight_layout()
    output_path = os.path.join(ARTIFACTS_DIR, "weight_histograms.png")
    plt.savefig(output_path)
    print(f"Histogram saved to: {output_path}")

    # Create a zoomed-in version for W1/W2 around zero
    plt.figure(figsize=(15, 5))
    
    # W1 Zoom
    data_w1 = joblib.load(os.path.join(DIR, FILES["W1 (base)"]))
    w1 = np.array(data_w1).flatten()
    plt.subplot(1, 2, 1)
    # Filter for values in [-1, 1] to see the sub-integer distribution
    w1_small = w1[np.abs(w1) < 1.0]
    plt.hist(w1_small, bins=100, color='salmon', edgecolor='black', alpha=0.7)
    plt.title("W1 Distribution (Zoomed |W| < 1.0)")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # W2 Zoom
    data_w2 = joblib.load(os.path.join(DIR, FILES["W2 (weights)"]))
    w2 = np.array(data_w2).flatten()
    plt.subplot(1, 2, 2)
    plt.hist(w2, bins=100, color='lightgreen', edgecolor='black', alpha=0.7)
    plt.title("W2 Distribution (Full Scale)")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    zoom_output_path = os.path.join(ARTIFACTS_DIR, "weight_histograms_zoom.png")
    plt.savefig(zoom_output_path)
    print(f"Zoomed histogram saved to: {zoom_output_path}")

if __name__ == "__main__":
    plot_histograms()
