import joblib
import numpy as np
import os

# Paths
DIR = r"c:\Users\jvald\HLS_Export\Weights\TPU_CODE"
BASE_PKL = os.path.join(DIR, "base.pkl") # W1
BIAS_PKL = os.path.join(DIR, "bias.pkl") # b1
WEIGHTS_PKL = os.path.join(DIR, "weights.pkl") # W2
BIAS2_PKL = os.path.join(DIR, "bias2.pkl") # b2
TEST_X_PKL = os.path.join(DIR, "X_test_scaled.pkl")
TEST_Y_PKL = os.path.join(DIR, "y_test.pkl")

HIDDEN_LIMIT = 1000
ANALYZE_FULL = True # Test 10,000 neurons if possible

def quantize(data, total_bits, int_bits):
    frac_bits = total_bits - int_bits
    scale = 2.0 ** frac_bits
    max_val = (2.0 ** (int_bits - 1)) - (1.0 / scale)
    min_val = -(2.0 ** (int_bits - 1))
    q = np.round(data * scale) / scale
    return np.clip(q, min_val, max_val)

def relu(x):
    return np.maximum(0, x)

def forward_pass(X, W1, b1, W2, b2):
    hid = np.dot(X, W1.T) + b1
    hid_act = relu(hid)
    out = np.dot(hid_act, W2.T) + b2
    return out

def get_accuracy(preds, labels):
    return np.mean(preds == labels)

def analyze_model(X, y, W1, b1, W2, b2, name):
    print(f"\n--- {name} ---")
    # Baseline
    logits_f = forward_pass(X, W1, b1, W2, b2)
    preds_f = np.argmax(logits_f, axis=1)
    acc_f = get_accuracy(preds_f, y)
    print(f"Baseline (Float32): {acc_f*100:.2f}%")

    # Sweep Fractional Bits for 4-bit width
    print("\n--- 4-bit Fractional Sweep ---")
    for ib in [2, 3, 4]:
        qW1 = quantize(W1, 4, ib)
        qb1 = quantize(b1, 4, ib)
        qW2 = quantize(W2, 4, ib)
        qb2 = quantize(b2, 4, ib)
        
        logits_q = forward_pass(X, qW1, qb1, qW2, qb2)
        preds_q = np.argmax(logits_q, axis=1)
        acc = get_accuracy(preds_q, y)
        
        # Calculate saturation % for W1 as an example
        sat_low = np.mean(W1 < -(2**(ib-1))) * 100
        sat_high = np.mean(W1 > (2**(ib-1) - 2**(ib-4))) * 100
        
        print(f"ap_fixed<4, {ib}> (I={ib}, F={4-ib}):")
        print(f"  Accuracy: {acc*100:.2f}%")
        print(f"  W1 Saturation: {sat_low + sat_high:.1f}%")

    # 8-bit
    qW1_8 = quantize(W1, 8, 4)
    qb1_8 = quantize(b1, 8, 4)
    qW2_8 = quantize(W2, 8, 4)
    qb2_8 = quantize(b2, 8, 4)
    logits_q8 = forward_pass(X, qW1_8, qb1_8, qW2_8, qb2_8)
    preds_q8 = np.argmax(logits_q8, axis=1)
    acc_q8 = get_accuracy(preds_q8, y)
    print(f"Comparison (8-bit): {acc_q8*100:.2f}%")
    print(f"  Accuracy Loss (8-bit): {(acc_f - acc_q8)*100:.2f} points")

def analyze():
    print("Loading Data...")
    X = joblib.load(TEST_X_PKL)
    y = joblib.load(TEST_Y_PKL)
    print(f"Labels distribution: {np.bincount(y)}")

    # Pruned Model
    W1_p = joblib.load(BASE_PKL)[:HIDDEN_LIMIT, :]
    b1_p = joblib.load(BIAS_PKL)[:HIDDEN_LIMIT]
    W2_p = joblib.load(WEIGHTS_PKL)[:, :HIDDEN_LIMIT]
    b2_p = joblib.load(BIAS2_PKL)
    analyze_model(X, y, W1_p, b1_p, W2_p, b2_p, f"Pruned Model ({HIDDEN_LIMIT} units)")

    if ANALYZE_FULL:
        W1_f = joblib.load(BASE_PKL)
        b1_f = joblib.load(BIAS_PKL)
        W2_f = joblib.load(WEIGHTS_PKL)
        b2_f = joblib.load(BIAS2_PKL)
        analyze_model(X, y, W1_f, b1_f, W2_f, b2_f, f"Full Model ({len(b1_f)} units)")

if __name__ == "__main__":
    analyze()
