import random

def generate_weights():
    N_INPUTS = 300
    N_HIDDEN = 10000
    N_OUTPUTS = 2
    
    with open('weights.h', 'w') as f:
        f.write('#ifndef WEIGHTS_H\n')
        f.write('#define WEIGHTS_H\n\n')
        f.write('#include <ap_fixed.h>\n\n')
        
        # Small dataset structure preserved for fallback/debug
        f.write('//#define USE_SMALL_DATASET // Uncomment to use small dataset for faster C-Sim/Synthesis debugging\n\n')
        f.write('#ifdef USE_SMALL_DATASET\n')
        
        # Calculate small dataset dimensions (10% of original total size)
        SMALL_INPUTS = N_INPUTS             # 300
        SMALL_HIDDEN = int(N_HIDDEN * 0.1)  # 1000
        SMALL_OUTPUTS = N_OUTPUTS           # 2

        f.write(f'#define N_INPUTS {SMALL_INPUTS}\n')
        f.write(f'#define N_HIDDEN {SMALL_HIDDEN}\n')
        f.write(f'#define N_OUTPUTS {SMALL_OUTPUTS}\n\n')
        
        # Using ap_fixed<16, 6> to accommodate values up to +/- 32 (like 2*pi per kernel.py)
        # 6 integer bits (1 sign + 5 magnitude) covers range [-32, 32)
        # 10 fractional bits provides precision ~0.001
        f.write('typedef ap_fixed<16, 6> weight_t;\n\n')
        
        # Generate dummy small weights
        f.write(f'static const weight_t W1[{SMALL_HIDDEN}][{SMALL_INPUTS}] = {{\n')
        row_str_small = '{' + ', '.join(['0.1'] * SMALL_INPUTS) + '}'
        for i in range(SMALL_HIDDEN):
            f.write('    ' + row_str_small + (',' if i < SMALL_HIDDEN - 1 else '') + '\n')
        f.write('};\n\n')
        
        f.write(f'static const weight_t b1[{SMALL_HIDDEN}] = {{')
        # Write b1 in chunks for readability if needed, but a single line might be too long for 500 elements
        # Let's break it up
        chunk_size = 50
        b1_vals = ['0.05'] * SMALL_HIDDEN
        f.write('\n')
        for i in range(0, SMALL_HIDDEN, chunk_size):
            chunk = b1_vals[i:i + chunk_size]
            f.write('    ' + ', '.join(chunk) + (',' if i + chunk_size < SMALL_HIDDEN else '') + '\n')
        f.write('};\n\n')
        
        f.write(f'static const weight_t W2[{SMALL_OUTPUTS}][{SMALL_HIDDEN}] = {{\n')
        row_w2_small = '{' + ', '.join(['0.1'] * SMALL_HIDDEN) + '}'
        for i in range(SMALL_OUTPUTS):
            f.write('    ' + row_w2_small + (',' if i < SMALL_OUTPUTS - 1 else '') + '\n')
        f.write('};\n\n')
        
        f.write(f'static const weight_t b2[{SMALL_OUTPUTS}] = {{')
        f.write(', '.join(['0.05'] * SMALL_OUTPUTS))
        f.write('};\n\n')
        
        f.write('#else\n\n')
        
        # Actual large dataset (Dummy values)
        f.write(f'#define N_INPUTS {N_INPUTS}\n')
        f.write(f'#define N_HIDDEN {N_HIDDEN}\n')
        f.write(f'#define N_OUTPUTS {N_OUTPUTS}\n\n')
        
        # Using ap_fixed<16, 6> for better precision and range [-32, 32]
        f.write('typedef ap_fixed<16, 6> weight_t;\n\n')
        
        # W1[N_HIDDEN][N_INPUTS]
        f.write(f'static const weight_t W1[{N_HIDDEN}][{N_INPUTS}] = {{\n')
        # To save space in the file, we can use a repeating pattern or zeros
        # But we need 5000 rows.
        # Writing 5000 rows x 300 cols to a file is ~1.5M floats.
        # Let's write them efficiently or maybe even compress logic if HLS allows,
        # but standardized array init is safest.
        # Just use 0.01 for everything to verify syntax quickly.
        row_str = '{' + ', '.join(['0.01'] * N_INPUTS) + '}'
        for i in range(N_HIDDEN):
             f.write('    ' + row_str + (',' if i < N_HIDDEN - 1 else '') + '\n')
        f.write('};\n\n')

        # b1[N_HIDDEN]
        f.write(f'static const weight_t b1[{N_HIDDEN}] = {{\n')
        # Write chunks to avoid super long line
        chunk_size = 100
        for i in range(0, N_HIDDEN, chunk_size):
            chunk = ['0.01'] * min(chunk_size, N_HIDDEN - i)
            f.write('    ' + ', '.join(chunk) + (',' if i + chunk_size < N_HIDDEN else '') + '\n')
        f.write('};\n\n')
        
        # W2[N_OUTPUTS][N_HIDDEN] = [2][5000]
        f.write(f'static const weight_t W2[{N_OUTPUTS}][{N_HIDDEN}] = {{\n')
        row_w2 = '{' + ', '.join(['0.01'] * N_HIDDEN) + '}'
        for i in range(N_OUTPUTS):
             f.write('    ' + row_w2 + (',' if i < N_OUTPUTS - 1 else '') + '\n')
        f.write('};\n\n')
        
        # b2[N_OUTPUTS] = [2]
        f.write(f'static const weight_t b2[{N_OUTPUTS}] = {{\n')
        f.write('    ' + ', '.join(['0.01'] * N_OUTPUTS) + '\n')
        f.write('};\n\n')
        
        f.write('#endif // USE_SMALL_DATASET\n\n')
        f.write('#endif // WEIGHTS_H\n')

if __name__ == '__main__':
    generate_weights()
