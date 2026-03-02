import joblib
import numpy as np
import os

# Paths
BASE_PKL = r"c:\Users\jvald\HLS_Export\Weights\TPU_CODE\base.pkl"
WEIGHTS_PKL = r"c:\Users\jvald\HLS_Export\Weights\TPU_CODE\weights.pkl"

def analyze_min_values(name, data):
    data_flat = data.flatten()
    non_zero = data_flat[data_flat != 0]
    
    if len(non_zero) == 0:
        print(f"\n--- Layer: {name} ---")
        print("All values are zero.")
        return

    abs_non_zero = np.abs(non_zero)
    min_val = np.min(abs_non_zero)
    max_val = np.max(abs_non_zero)
    
    print(f"\n--- Layer: {name} ---")
    print(f"Number of non-zero weights: {len(non_zero)}")
    print(f"Minimum absolute non-zero value: {min_val:.12f}")
    print(f"Maximum absolute value: {max_val:.12f}")
    
    # Required fractional bits to represent the smallest value (at least 1 bit of precision for that value)
    # value = 2^-frac_bits => frac_bits = -log2(value)
    required_frac_bits = int(np.ceil(-np.log2(min_val)))
    print(f"Required fractional bits to represent min value: {required_frac_bits}")
    
    # Distribution of magnitudes
    magnitudes = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    for mag in magnitudes:
        count = np.sum(abs_non_zero < mag)
        print(f"Values smaller than {mag}: {count} ({count/len(data_flat)*100:.4f}%)")

def main():
    if not os.path.exists(BASE_PKL) or not os.path.exists(WEIGHTS_PKL):
        print("Error: Pickle files not found.")
        return

    print("Loading weights...")
    W1 = np.array(joblib.load(BASE_PKL))
    W2 = np.array(joblib.load(WEIGHTS_PKL))
    
    analyze_min_values("W1 (Hidden weights)", W1)
    analyze_min_values("W2 (Output weights)", W2)

if __name__ == "__main__":
    main()
