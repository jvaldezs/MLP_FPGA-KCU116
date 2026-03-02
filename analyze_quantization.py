import joblib
import numpy as np
import os

# Paths
BASE_PKL = r"c:\Users\jvald\HLS_Export\Weights\TPU_CODE\base.pkl"
WEIGHTS_PKL = r"c:\Users\jvald\HLS_Export\Weights\TPU_CODE\weights.pkl"

def quantize_ap_fixed(data, total_bits=16, int_bits=6):
    frac_bits = total_bits - int_bits
    scale = 2.0 ** frac_bits
    
    max_val = (2.0 ** (int_bits - 1)) - (1.0 / scale)
    min_val = -(2.0 ** (int_bits - 1))
    
    q_data = np.round(data * scale) / scale
    clipped_data = np.clip(q_data, min_val, max_val)
    
    return clipped_data

def analyze_layer(name, data, configs):
    print(f"\n--- Layer: {name} ({len(data)} weights) ---")
    data_flat = data.flatten()
    orig_zeros = np.sum(data_flat == 0)
    print(f"Original Zeros: {orig_zeros} ({orig_zeros/len(data_flat)*100:.4f}%)")
    
    for tb, ib in configs:
        q_data = quantize_ap_fixed(data_flat, tb, ib)
        q_zeros = np.sum(q_data == 0)
        newly_rounded_to_zero = q_zeros - orig_zeros
        
        error = data_flat - q_data
        mae = np.mean(np.abs(error))
        
        print(f"ap_fixed<{tb}, {ib}>: Total Zeros = {q_zeros} | Newly Rounded = {newly_rounded_to_zero} ({newly_rounded_to_zero/len(data_flat)*100:.4f}%) | MAE = {mae:.8f}")

def main():
    if not os.path.exists(BASE_PKL) or not os.path.exists(WEIGHTS_PKL):
        print("Error: Pickle files not found.")
        return

    print("Loading weights...")
    W1 = np.array(joblib.load(BASE_PKL))
    W2 = np.array(joblib.load(WEIGHTS_PKL))
    
    configs = [
        (16, 4), # Smaller integer range, more precision
        (16, 6), # Balanced
        (16, 8)  # Larger integer range, less precision
    ]
    
    analyze_layer("W1 (Hidden weights)", W1, configs)
    analyze_layer("W2 (Output weights)", W2, configs)

if __name__ == "__main__":
    main()
