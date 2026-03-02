import joblib
import numpy as np
import os
import sys

# Paths - mapping PKL files to their model names
DIR = r"c:\Users\jvald\HLS_Export\Weights\TPU_CODE"
FILES = {
    "on_w1": "base.pkl",
    "on_b1": "bias.pkl",
    "on_w2": "weights.pkl",
    "on_b2": "bias2.pkl"
}

def find_nearest_zero(matrix_key):
    filename = FILES.get(matrix_key)
    if not filename:
        print(f"Unknown matrix key: {matrix_key}")
        return None

    path = os.path.join(DIR, filename)
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return None

    print(f"--- Nearest-to-Zero Analysis of {matrix_key} ({filename}) ---")
    
    # Load the data
    data = joblib.load(path)
    arr = np.array(data).flatten()
    
    # We want to find the smallest non-zero absolute value
    # Initialize with infinity
    min_abs_val = float('inf')
    found_val = 0.0
    
    # Manual loop as requested
    for j in range(len(arr)):
        val = arr[j]
        abs_val = abs(val)
        
        # We ignore pure zeros for this specific analysis 
        # (unless the user wants to see how close we get to actual zero)
        if abs_val > 0 and abs_val < min_abs_val:
            min_abs_val = abs_val
            found_val = val
            
    if min_abs_val == float('inf'):
        print(f"  All values in this matrix are exactly 0.0")
        return {"key": matrix_key, "val": 0.0, "abs_val": 0.0}
    else:
        print(f"  Smallest non-zero value: {found_val:.12f}")
        print(f"  Absolute distance to zero: {min_abs_val:.12f}")
        print("-" * 40)
        return {"key": matrix_key, "val": found_val, "abs_val": min_abs_val}

if __name__ == "__main__":
    results = []
    for key in FILES.keys():
        res = find_nearest_zero(key)
        if res:
            results.append(res)
    
    # Export to file
    with open("distance_from_zero.txt", "w") as f:
        f.write("Matrix,Smallest_Value,Distance_to_Zero\n")
        for r in results:
            f.write(f"{r['key']},{r['val']:.12f},{r['abs_val']:.12f}\n")
    
    print(f"\nResults exported to: {os.path.abspath('distance_from_zero.txt')}")
