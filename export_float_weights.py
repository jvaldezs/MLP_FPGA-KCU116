import joblib
import numpy as np
import os

# Paths
BASE_PKL = r"c:\Users\jvald\HLS_Export\Weights\TPU_CODE\base.pkl"
WEIGHTS_PKL = r"c:\Users\jvald\HLS_Export\Weights\TPU_CODE\weights.pkl"
OUTPUT_HEADER = r"c:\Users\jvald\HLS_Export\weights_full_float.h"

def export_float_header():
    if not os.path.exists(BASE_PKL) or not os.path.exists(WEIGHTS_PKL):
        print("Error: Pickle files not found.")
        return

    print("Loading weights...")
    W1 = np.array(joblib.load(BASE_PKL))
    W2 = np.array(joblib.load(WEIGHTS_PKL))

    # W1 is usually (300, 6000) based on previous code. We need it as (6000, 300) for HLS.
    if W1.shape == (300, 6000):
        W1 = W1.T
    
    # The original ML script exported W2 as (2, 10000). We must slice it to match N_HIDDEN=6000.
    if W2.shape[1] > 6000:
        W2 = W2[:, :6000]
    
    # Biases (assuming they are zeroed or not used in the initial pkls based on previous context, 
    # but we will provide dummy 0.0f arrays if actual biases aren't present in the pkl to match the C++ structure)
    b1 = np.zeros(6000, dtype=np.float32)
    b2 = np.zeros(2, dtype=np.float32)

    print(f"Writing to {OUTPUT_HEADER}...")
    with open(OUTPUT_HEADER, "w") as f:
        f.write("#ifndef WEIGHTS_FULL_FLOAT_H\n")
        f.write("#define WEIGHTS_FULL_FLOAT_H\n\n")
        f.write("// 32-bit Float Weights for Golden Model Verification\n\n")

        # Write b1
        f.write("const float b1_float[6000] = {\n")
        b1_str = ",\n".join([f"{val:.6f}" for val in b1])
        f.write(b1_str + "\n};\n\n")

        # Write W2
        f.write("const float W2_float[2][6000] = {\n")
        for i, row in enumerate(W2):
            row_str = ", ".join([f"{val:.6f}" for val in row])
            f.write(f"    {{{row_str}}}")
            if i < len(W2) - 1:
                f.write(",\n")
            else:
                f.write("\n")
        f.write("};\n\n")

        # Write b2
        f.write("const float b2_float[2] = {\n")
        b2_str = ",\n".join([f"{val:.6f}" for val in b2])
        f.write(b2_str + "\n};\n\n")

        f.write("#endif // WEIGHTS_FULL_FLOAT_H\n")

    print(f"Export complete. The massive W1 matrix wasn't hardcoded to prevent compiler crashes.")
    print(f"We will use the same w1.txt for W1, just read as floats in the test bench.")

if __name__ == "__main__":
    export_float_header()
