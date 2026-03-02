import joblib
import numpy as np
import os
import sys

# Paths from Reproduction
# We use the files we just generated in Weights/TPU_CODE
DIR = r"c:\Users\jvald\HLS_Export\Weights\TPU_CODE"
BASE_PKL = os.path.join(DIR, "base.pkl") # W1
BIAS_PKL = os.path.join(DIR, "bias.pkl") # b1
WEIGHTS_PKL = os.path.join(DIR, "weights.pkl") # W2
BIAS2_PKL = os.path.join(DIR, "bias2.pkl") # b2
TEST_X_PKL = os.path.join(DIR, "X_test_scaled.pkl")
TEST_Y_PKL = os.path.join(DIR, "y_test.pkl")

def quantize(data, total_bits, int_bits=6):
    frac_bits = total_bits - int_bits
    scale = 2.0 ** frac_bits
    max_val = (2.0 ** (int_bits - 1)) - (1.0 / scale)
    min_val = -(2.0 ** (int_bits - 1))
    q = np.round(data * scale) / scale
    return np.clip(q, min_val, max_val)

def relu(x):
    return np.maximum(0, x)

def forward_pass(X, W1, b1, W2, b2):
    # W1: (D, 300) -> Need W1.T for dot(X, W.T) if X is (N, 300)
    # The W1 saved was (D, nFeatures) = (3000, 300).
    # X: (N, 300).
    # X @ W1.T -> (N, 3000).
    
    # Layer 1
    # Note: If W1 comes from joblib, it is ndarray.
    hid = np.dot(X, W1.T) + b1
    hid_act = relu(hid)
    
    # Layer 2
    # W2: (2, 3000)
    # Out: (N, 2)
    out = np.dot(hid_act, W2.T) + b2
    return out

def get_accuracy(preds, labels):
    return np.mean(preds == labels)

def analyze():
    print("Loading Data & Weights...")
    try:
        X = joblib.load(TEST_X_PKL)
        y = joblib.load(TEST_Y_PKL)
        W1 = joblib.load(BASE_PKL)
        b1 = joblib.load(BIAS_PKL)
        W2 = joblib.load(WEIGHTS_PKL)
        b2 = joblib.load(BIAS2_PKL)
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    print(f"Test Data: {X.shape}")
    print(f"W1: {W1.shape}, b1: {b1.shape}")
    print(f"W2: {W2.shape}, b2: {b2.shape}")
    
    print("Baseline (Float32)...")
    logits = forward_pass(X, W1, b1, W2, b2)
    preds = np.argmax(logits, axis=1)
    baseline_acc = get_accuracy(preds, y)
    print(f"Baseline Accuracy: {baseline_acc*100:.2f}%")
    
    threshold = baseline_acc - 0.05
    print(f"Threshold (5% loss): {threshold*100:.2f}%")
    
    int_bits = 6
    bits_to_test = [32, 16, 14, 12, 10, 8, 6, 4]
    
    for b in bits_to_test:
        print(f"\nTesting ap_fixed<{b}, {int_bits}>...")
        
        qW1 = quantize(W1, b, int_bits)
        qb1 = quantize(b1, b, int_bits)
        qW2 = quantize(W2, b, int_bits)
        qb2 = quantize(b2, b, int_bits)
        
        # Quantize Input too? 
        # HLS usually accepts 'fixed point' inputs if interface is fixed.
        # But if inputs come from outside as float, they get cast. 
        # For strict simulation, yes.
        qX = quantize(X, b, int_bits)
        
        logits_q = forward_pass(qX, qW1, qb1, qW2, qb2)
        preds_q = np.argmax(logits_q, axis=1)
        acc = get_accuracy(preds_q, y)
        
        print(f"Accuracy: {acc*100:.2f}%")
        
        if acc <= threshold:
            print(f"!! DROP DETECTED !! Accuracy dropped below threshold at {b} bits.")
            # Don't break immediately, let's see how bad it gets.
    
    print("\nAnalysis Complete.")

if __name__ == "__main__":
    analyze()
