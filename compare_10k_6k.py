import numpy as np
import math
from sklearn.linear_model import RidgeClassifier
import os

def load_data_simple():
    # Bypassing pandas using numpy
    train_data = np.genfromtxt('Weights/TPU_CODE/filtered_training.csv', delimiter=',', skip_header=1)
    test_data = np.genfromtxt('Weights/TPU_CODE/filtered_testing.csv', delimiter=',', skip_header=1)
    
    # Simple windowing logic mimic
    window_size = 300
    overlap = 0.95
    step = int(window_size * (1 - overlap))
    
    def process(data_raw, fault_start, fault_end):
        times = data_raw[:, 0]
        currents = data_raw[:, 1:]
        X, y = [], []
        for i in range(0, len(data_raw) - window_size, step):
            window_times = times[i:i+window_size]
            window_currents = currents[i:i+window_size].flatten()
            label = 1 if any(fault_start <= t <= fault_end for t in window_times) else 0
            X.append(window_currents)
            y.append(label)
        return np.array(X), np.array(y)

    X_train, y_train = process(train_data, 9.0, 9.15)
    X_test, y_test = process(test_data, 9.03, 9.1)
    
    # Scale
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    
    return X_train, y_train, X_test, y_test

def compare_scales():
    print("Loading data...")
    X_train, y_train, X_test, y_test = load_data_simple()
    nFeatures = X_train.shape[1]
    
    results = {}
    for D in [10000, 6000]:
        print(f"Testing D={D}...")
        # Fix seed for reproducibility
        np.random.seed(42)
        W1 = np.random.normal(0, 1, (D, nFeatures))
        b1 = np.random.uniform(0, 2*math.pi, D)
        
        # Layer 1
        X_train_enc = np.maximum(0, np.dot(X_train, W1.T) + b1)
        X_test_enc = np.maximum(0, np.dot(X_test, W1.T) + b1)
        
        # Layer 2 (Fixed 16-bit proxy for quantization effect)
        # We calculate Float32 accuracy first
        clf = RidgeClassifier(alpha=1.0)
        clf.fit(X_train_enc, y_train)
        acc_float = clf.score(X_test_enc, y_test)
        
        # Simulation of 16-bit weight mismatch (relative ~0.4%)
        # Just to see the combined effect
        W1_fixed = np.round(W1 * 4096) / 4096 # 12 fractional bits
        X_train_fixed = np.maximum(0, np.dot(X_train, W1_fixed.T) + b1)
        X_test_fixed = np.maximum(0, np.dot(X_test, W1_fixed.T) + b1)
        clf_fixed = RidgeClassifier(alpha=1.0)
        clf_fixed.fit(X_train_fixed, y_train)
        acc_fixed = clf_fixed.score(X_test_fixed, y_test)
        
        results[D] = (acc_float, acc_fixed)
        print(f"  D={D}: Float Acc={acc_float*100:.2f}%, Fixed16 Acc={acc_fixed*100:.2f}%")

    diff = results[10000][0] - results[6000][1]
    print(f"\nTOTAL MARGIN (10k Float -> 6k Fixed16): {diff*100:.4f}%")

if __name__ == "__main__":
    compare_scales()
