import joblib
import numpy as np
import os

DIR = r"c:\Users\jvald\HLS_Export\Weights\TPU_CODE"
BASE_PKL = os.path.join(DIR, "base.pkl")
BIAS_PKL = os.path.join(DIR, "bias.pkl")
WEIGHTS_PKL = os.path.join(DIR, "weights.pkl")
BIAS2_PKL = os.path.join(DIR, "bias2.pkl")

def export_banked_weights(prefix, hidden_limit, type_str):
    header_file = f"{prefix}.h"
    tcl_file = "run_optimized.tcl"
    
    # 1. Load Data
    W1_all = joblib.load(BASE_PKL)
    b1_all = joblib.load(BIAS_PKL)
    W2_all = joblib.load(WEIGHTS_PKL)
    b2_all = joblib.load(BIAS2_PKL)
    
    W1 = W1_all[:hidden_limit, :]
    b1 = b1_all[:hidden_limit]
    W2 = W2_all[:, :hidden_limit]
    b2 = b2_all
    
    N_HIDDEN, N_INPUTS = W1.shape
    N_OUTPUTS = W2.shape[0]
    
    N_URAM_HIDDEN = 3400
    N_BRAM_HIDDEN = hidden_limit - N_URAM_HIDDEN
    
    W1_URAM = W1[:N_URAM_HIDDEN, :]
    W1_BRAM = W1[N_URAM_HIDDEN:, :]

    # Export entire W1 to a raw text file for the testbench to read
    print("Exporting w1.txt for testbench...")
    with open("w1.txt", "w") as f:
        for row in W1:
            f.write(" ".join(f"{x:.6f}" for x in row) + "\n")
    
    cpp_files = [f"{prefix}_other.cpp"]

    # Export b1, W2, b2
    with open(cpp_files[0], 'w') as f:
        f.write(f'#include "{header_file}"\n\n')
        
        # b1
        f.write(f'extern const weight_t b1[{N_HIDDEN}] = {{\n')
        for i in range(0, N_HIDDEN, 50):
            chunk = b1[i:i+50]
            f.write(f'    {", ".join(f"{x:.6f}" for x in chunk)},\n')
        f.write('};\n\n')
        
        # W2
        f.write(f'extern const weight_t W2[{N_OUTPUTS}][{N_HIDDEN}] = {{\n')
        for i in range(N_OUTPUTS):
            row_str = ', '.join(f'{x:.6f}' for x in W2[i])
            f.write(f'    {{{row_str}}},\n')
        f.write('};\n\n')
        
        # b2
        f.write(f'extern const weight_t b2[{N_OUTPUTS}] = {{\n')
        f.write(f'    {", ".join(f"{x:.6f}" for x in b2)}\n')
        f.write('};\n')
        
    print(f"Exported {cpp_files[0]}")

    guard = prefix.upper().replace('.', '_') + "_H"

    # Generate Header
    with open(header_file, 'w') as f:
        f.write(f'#ifndef {guard}\n')
        f.write(f'#define {guard}\n\n')
        f.write(f'#include "ap_fixed.h"\n\n')
        f.write(f'#define N_INPUTS {N_INPUTS}\n')
        f.write(f'#define N_HIDDEN {N_HIDDEN}\n')
        f.write(f'#define N_URAM_HIDDEN {N_URAM_HIDDEN}\n')
        f.write(f'#define N_BRAM_HIDDEN {N_BRAM_HIDDEN}\n')
        f.write(f'#define N_OUTPUTS {N_OUTPUTS}\n\n')
        f.write(f'typedef {type_str} weight_t;\n\n')
        f.write(f'extern const weight_t b1[{N_HIDDEN}];\n')
        f.write(f'extern const weight_t W2[{N_OUTPUTS}][{N_HIDDEN}];\n')
        f.write(f'extern const weight_t b2[{N_OUTPUTS}];\n\n')
        f.write('#endif\n')

    # Update run_optimized.tcl
    with open(tcl_file, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    added_sources = False
    for line in lines:
        if line.startswith("add_files") and (".cpp" in line) and ("-tb" not in line):
            if not added_sources:
                new_lines.append(f'add_files mlp_optimized.cpp -cflags "-std=c++11"\n')
                for cpp in cpp_files:
                    new_lines.append(f'add_files {cpp} -cflags "-std=c++11"\n')
                added_sources = True
            else:
                continue # Skip other design files
        else:
            new_lines.append(line)
            
    with open(tcl_file, 'w') as f:
        f.writelines(new_lines)
    print(f"Updated {tcl_file}")

if __name__ == "__main__":
    export_banked_weights("weights_small", 6000, "ap_fixed<16, 4>")
    print("Done. Runtime weights generated successfully.")
