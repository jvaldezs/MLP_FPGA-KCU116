import joblib
import numpy as np
import os
import csv

# Paths
BASE_PKL = r"c:\Users\jvald\HLS_Export\Weights\TPU_CODE\base.pkl"
WEIGHTS_PKL = r"c:\Users\jvald\HLS_Export\Weights\TPU_CODE\weights.pkl"
OUTPUT_CSV = r"c:\Users\jvald\HLS_Export\quantization_summary.csv"

def run_analysis():
    if not os.path.exists(BASE_PKL) or not os.path.exists(WEIGHTS_PKL):
        print("Error: Pickle files not found.")
        return

    print("Loading weights...")
    W1 = np.array(joblib.load(BASE_PKL)).flatten()
    W2 = np.array(joblib.load(WEIGHTS_PKL)).flatten()

    def get_rounding_stats(data, frac_bits):
        threshold = 2.0 ** -(frac_bits + 1)
        total = len(data)
        newly_rounded = np.sum((np.abs(data) < threshold) & (data != 0))
        pct = (newly_rounded / total) * 100
        return newly_rounded, pct

    # Formats to test
    formats = [
        {"name": "ap_fixed<16, 4>", "frac_bits": 12},
        {"name": "ap_fixed<8, 4>",  "frac_bits": 4},
        {"name": "ap_fixed<4, 4>",  "frac_bits": 0} # 4-bit with 4 int bits = 0 frac bits
    ]

    with open(OUTPUT_CSV, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Format", "Layer", "Total Weights", "Newly Rounded to Zero", "Rounding Loss (%)"])
        
        for fmt in formats:
            # W1
            w1_round, w1_pct = get_rounding_stats(W1, fmt["frac_bits"])
            writer.writerow([fmt["name"], "W1", len(W1), w1_round, f"{w1_pct:.10f}"])
            
            # W2
            w2_round, w2_pct = get_rounding_stats(W2, fmt["frac_bits"])
            writer.writerow([fmt["name"], "W2", len(W2), w2_round, f"{w2_pct:.10f}"])

    print(f"Summary exported to: {OUTPUT_CSV}")

if __name__ == "__main__":
    run_analysis()
