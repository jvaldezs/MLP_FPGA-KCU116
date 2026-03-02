import numpy as np
import matplotlib.pyplot as plt

def analyze_mismatch(D=6000, nFeatures=300):
    print(f"--- Analyzing 16-bit Quantization Mismatch (N_HIDDEN={D}) ---")
    
    # 1. Generate Full-Precision Weights (Float32)
    # Using a normal distribution as a proxy for trained weights
    W_float = np.random.normal(0, 0.1, (D, nFeatures)).astype(np.float32)
    
    # 2. Simulate 16-bit Fixed-Point (ap_fixed<16, 4>)
    # ap_fixed<16, 4> has 4 integer bits (including sign) and 12 fractional bits.
    # Step size (LSB) = 2^-12
    lsb = 2**-12
    range_max = (2**3) - lsb # 4 bits total: 1 sign, 3 integer
    range_min = -(2**3)
    
    W_fixed = np.round(W_float / lsb) * lsb
    W_fixed = np.clip(W_fixed, range_min, range_max)
    
    # 3. Mismatch Analysis
    diff = W_float - W_fixed
    abs_diff = np.abs(diff)
    
    print(f"Max Absolute Error: {np.max(abs_diff):.8f}")
    print(f"Mean Absolute Error: {np.mean(abs_diff):.8f}")
    print(f"RMS Error: {np.sqrt(np.mean(diff**2)):.8f}")
    
    # Rounded to Zero Analysis
    n_zeros_float = np.sum(W_float == 0)
    n_zeros_fixed = np.sum(W_fixed == 0)
    newly_zeroed = np.sum((W_float != 0) & (W_fixed == 0))
    total_elements = D * nFeatures
    
    print(f"\nZero Analysis:")
    print(f"  Total Elements: {total_elements:,}")
    print(f"  Float Zeros:    {n_zeros_float}")
    print(f"  Fixed Zeros:    {n_zeros_fixed}")
    print(f"  Newly Zeroed:   {newly_zeroed} ({newly_zeroed/total_elements*100:.4f}%)")
    
    # Relative Error on Non-Zero Elements
    non_zero_mask = (W_float != 0)
    rel_error = abs_diff[non_zero_mask] / np.abs(W_float[non_zero_mask])
    print(f"\nRelative Error (Non-Zero Elements):")
    print(f"  Mean Relative Error: {np.mean(rel_error)*100:.4f}%")
    print(f"  Max Relative Error:  {np.max(rel_error)*100:.4f}%")

    # 4. Activation Mismatch Simulation
    # Simulate a dot product with a random input[-1, 1]
    input_vec = np.random.uniform(-1, 1, nFeatures).astype(np.float32)
    
    act_float = np.dot(W_float, input_vec)
    act_fixed = np.dot(W_fixed, input_vec)
    
    act_diff = np.abs(act_float - act_fixed)
    print(f"\nActivation Mismatch (Dot Product with random input):")
    print(f"  Mean Difference: {np.mean(act_diff):.6f}")
    print(f"  Max Difference:  {np.max(act_diff):.6f}")
    print(f"  Relative Diff:   {np.mean(act_diff/np.abs(act_float))*100:.4f}%")

    # Plot Distribution of Errors
    plt.figure(figsize=(10, 6))
    plt.hist(diff.flatten(), bins=100, color='skyblue', edgecolor='black')
    plt.title(f"Weight Quantization Error Distribution (16-bit, N_HIDDEN={D})")
    plt.xlabel("Error (Float32 - Fixed16)")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.savefig('mismatch_analysis_6000.png')
    print("\nError distribution plot saved as 'mismatch_analysis_6000.png'")

if __name__ == "__main__":
    analyze_mismatch()
