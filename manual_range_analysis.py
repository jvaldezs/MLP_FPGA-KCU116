import joblib
import numpy as np
import os
import sys
import argparse

# Paths - mapping PKL files to their model names
DIR = r"c:\Users\jvald\HLS_Export\Weights\TPU_CODE"
FILES = {
    "on_w1": "base.pkl",
    "on_b1": "bias.pkl",
    "on_w2": "weights.pkl",
    "on_b2": "bias2.pkl"
}

def manual_analyze(matrix_key):
    filename = FILES.get(matrix_key)
    if not filename:
        print(f"Unknown matrix key: {matrix_key}")
        return

    path = os.path.join(DIR, filename)
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    print(f"--- Manual Analysis of {matrix_key} ({filename}) ---")
    
    # Load the data
    data = joblib.load(path)
    
    # Ensure it's a flattened numpy array for simple iterative loop
    arr = np.array(data).flatten()
    
    # Initialize min/max using the first element
    min_val = arr[0]
    max_val = arr[0]
    
    # Manual loop as requested with IF statement
    for j in range(len(arr)):
        current_val = arr[j]
        
        # Check for minimum
        if current_val < min_val:
            min_val = current_val
            
        # Check for maximum
        if current_val > max_val:
            max_val = current_val
            
    print(f"  Count: {len(arr)}")
    print(f"  Min Value: {min_val:.8f}")
    print(f"  Max Value: {max_val:.8f}")
    print(f"  Range: {max_val - min_val:.8f}")
    print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual Weight Range Analysis")
    parser.add_argument("--matrix", type=str, help="Matrix to analyze: on_w1, on_b1, on_w2, on_b2 (default: all)")
    
    args = parser.parse_args()
    
    if args.matrix:
        manual_analyze(args.matrix)
    else:
        for key in FILES.keys():
            manual_analyze(key)
