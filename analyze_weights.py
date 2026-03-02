import re

def analyze_weights(filename):
    max_val = -float('inf')
    min_val = float('inf')
    
    with open(filename, 'r') as f:
        content = f.read()
        # Find all floating point numbers
        # This regex looks for numbers like -0.123, 0.123, 1.23e-05
        numbers = re.findall(r'-?\d+\.\d+(?:e-?\d+)?', content)
        
        print(f"Found {len(numbers)} numbers")
        
        for num_str in numbers:
            try:
                val = float(num_str)
                if val > max_val: max_val = val
                if val < min_val: min_val = val
            except ValueError:
                continue
                
    print(f"Max value: {max_val}")
    print(f"Min value: {min_val}")
    
    max_abs = max(abs(max_val), abs(min_val))
    print(f"Max absolute value: {max_abs}")
    
    # Calculate required integer bits for ap_fixed
    # ap_fixed<W, I>
    # Bits for integer part (including sign) depends on range.
    # range [-2^(I-1), 2^(I-1) - step]
    
    import math
    if max_abs > 0:
        int_bits = math.ceil(math.log2(max_abs)) + 1 # +1 for sign
        if int_bits < 1: int_bits = 1 # At least sign bit
    else:
        int_bits = 1
        
    print(f"Suggested Integer Bits (I) for ap_fixed<W, I>: {int_bits} (assuming signed)")

if __name__ == "__main__":
    analyze_weights(r"e:\FPGA_application\FPGA_application\HLS_Export\weights.h")
