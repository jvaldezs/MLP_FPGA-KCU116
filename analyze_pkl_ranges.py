import joblib
import numpy as np
import os
import math

DIR = r"c:\Users\jvald\HLS_Export\Weights\TPU_CODE"
FILES = ["base.pkl", "bias.pkl", "weights.pkl", "bias2.pkl"]

def analyze_pkl():
    with open("weight_analysis.log", "w") as out:
        for filename in FILES:
            path = os.path.join(DIR, filename)
            if not os.path.exists(path):
                out.write(f"Skipping {filename} - not found\n")
                continue
                
            data = joblib.load(path)
            # Convert to numpy if not already
            arr = np.array(data)
            
            min_v = np.min(arr)
            max_v = np.max(arr)
            max_abs = np.max(np.abs(arr))
            
            out.write(f"--- {filename} ---\n")
            out.write(f"  Min: {min_v:.6f}\n")
            out.write(f"  Max: {max_v:.6f}\n")
            out.write(f"  Max Abs: {max_abs:.6f}\n")
            
            if max_abs > 0:
                int_bits = math.ceil(math.log2(max_abs)) + 1
                if int_bits < 1: int_bits = 1
            else:
                int_bits = 1
                
            out.write(f"  Required Integer Bits (signed): {int_bits}\n")
            
            if int_bits >= 4:
                out.write(f"  WARNING: 4-bit width (ap_fixed<4,{int_bits}>) will cause significant overflow/saturation!\n")
            else:
                frac_bits = 4 - int_bits
                out.write(f"  If using 4-bit total width: I={int_bits}, F={frac_bits} (ap_fixed<4, {int_bits}>)\n")
            out.write("\n")

if __name__ == "__main__":
    analyze_pkl()
