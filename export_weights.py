import joblib
import numpy as np
import os
import sys

# Paths
DIR = r"c:\Users\jvald\HLS_Export\Weights\TPU_CODE"
BASE_PKL = os.path.join(DIR, "base.pkl") # W1
BIAS_PKL = os.path.join(DIR, "bias.pkl") # b1
WEIGHTS_PKL = os.path.join(DIR, "weights.pkl") # W2
BIAS2_PKL = os.path.join(DIR, "bias2.pkl") # b2
OUTPUT_FULL = "weights.h"
OUTPUT_SMALL = "weights_small.h" # Still nice to keep small fallback

# HLS Type
# Reverting to float32 for accuracy
TYPE_STR = "float"

def export_weights():
    if not os.path.exists(BASE_PKL):
        print(f"Error: {BASE_PKL} not found.")
        return

    print("Loading weights...")
    W1 = joblib.load(BASE_PKL) # (3000, 300) or (10000, 300)
    b1 = joblib.load(BIAS_PKL) # (3000,)
    W2 = joblib.load(WEIGHTS_PKL) # (2, 3000)
    b2 = joblib.load(BIAS2_PKL) # (2,)
    
    # HLS expects W1[N_HIDDEN][N_INPUTS]
    # Loaded W1 is (N_HIDDEN, N_INPUTS).
    # If W1 was (N_INPUTS, N_HIDDEN), we would transpose.
    # Checks:
    if W1.shape[1] > W1.shape[0] and W1.shape[1] >= 1000:
       # Likely (300, 10000). Transpose.
       # Wait, reproduced training saves as (D, nFeatures) = (10000, 300).
       # So W1[10000][300] is correct for C array 'W1[10000][300]'.
       pass
    
    print(f"W1 Shape: {W1.shape}")
    print(f"b1 Shape: {b1.shape}")
    print(f"W2 Shape: {W2.shape}")
    print(f"b2 Shape: {b2.shape}")
    
    N_HIDDEN, N_INPUTS = W1.shape
    N_OUTPUTS = W2.shape[0]

    with open(OUTPUT_FULL, 'w') as f:
        f.write('#ifndef WEIGHTS_H\n')
        f.write('#define WEIGHTS_H\n\n')
        
        f.write(f'#define N_INPUTS {N_INPUTS}\n')
        f.write(f'#define N_HIDDEN {N_HIDDEN}\n')
        f.write(f'#define N_OUTPUTS {N_OUTPUTS}\n\n')
        
        f.write(f'typedef {TYPE_STR} weight_t;\n\n')
        
        # W1
        f.write(f'static const weight_t W1[{N_HIDDEN}][{N_INPUTS}] = {{\n')
        for i in range(N_HIDDEN):
            row = ', '.join(f'{x:.6f}' for x in W1[i])
            f.write(f'    {{{row}}},\n')
        f.write('};\n\n')
        
        # b1
        f.write(f'static const weight_t b1[{N_HIDDEN}] = {{\n')
        # Write flat array
        chunk_size = 50
        for i in range(0, N_HIDDEN, chunk_size):
            chunk = b1[i:i+chunk_size]
            row = ', '.join(f'{x:.6f}' for x in chunk)
            f.write(f'    {row},\n')
        f.write('};\n\n')
        
        # W2
        f.write(f'static const weight_t W2[{N_OUTPUTS}][{N_HIDDEN}] = {{\n')
        for i in range(N_OUTPUTS):
            row = ', '.join(f'{x:.6f}' for x in W2[i])
            f.write(f'    {{{row}}},\n')
        f.write('};\n\n')
        
        # b2
        f.write(f'static const weight_t b2[{N_OUTPUTS}] = {{\n')
        row = ', '.join(f'{x:.6f}' for x in b2)
        f.write(f'    {row}\n')
        f.write('};\n\n')
        
        f.write('#endif // WEIGHTS_H\n')
    
    print("Export Complete.")

if __name__ == "__main__":
    export_weights()
