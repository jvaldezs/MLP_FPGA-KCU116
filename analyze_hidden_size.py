import sys
import os

# Add Weights/TPU_CODE to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'Weights', 'TPU_CODE'))

import Config
import KernelFunctions
import numpy as np
import joblib
import math

# Override Config
Config.directory = r"c:\Users\jvald\HLS_Export\Weights\TPU_CODE\\"
Config.mu = 0
Config.sigma = 1

def run_sweep():
    print("Loading Data...")
    original_cwd = os.getcwd()
    
    # Change to Weights/TPU_CODE so KernelFunctions can find CSVs
    target_dir = os.path.join(original_cwd, 'Weights', 'TPU_CODE')
    os.chdir(target_dir)
    
    try:
        X_train, y_train, X_test, y_test, nFeatures, nClasses = KernelFunctions.load("","")
    except Exception as e:
        print(f"Error loading data: {e}")
        os.chdir(original_cwd)
        return
    finally:
        os.chdir(original_cwd)

    print(f"Train Scale: Mean {X_train.mean():.4f}, Std {X_train.std():.4f}")
    print(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")
    
    # Sweep D values
    # User uses 10000. Small dataset was 1000.
    d_values = [1000, 2000, 3000, 5000, 7500, 10000]
    
    results = {}
    
    for D in d_values:
        print(f"\n--- Testing Hidden Dimension D = {D} ---")
        Config.D = D
        
        # 1. Generate W1 (Base)
        W1 = np.random.normal(0, 1, (D, nFeatures))
        b1 = np.random.uniform(0, 2*math.pi, D)
        
        # 2. Encode (Layer 1) with ReLU
        X_train_enc = np.dot(X_train, W1.T) + b1
        X_train_enc = np.maximum(0, X_train_enc)
        
        X_test_enc = np.dot(X_test, W1.T) + b1
        X_test_enc = np.maximum(0, X_test_enc)
        
        # 3. Train Classifier (Layer 2)
        from sklearn.linear_model import RidgeClassifier
        clf = RidgeClassifier(alpha=1.0)
        clf.fit(X_train_enc, y_train)
        
        acc = clf.score(X_test_enc, y_test)
        print(f"Result for D={D}: Accuracy = {acc*100:.2f}%")
        results[D] = acc
        
    print("\n--- Summary ---")
    print("Dimension (D) | Accuracy")
    print("--------------|---------")
    for D in d_values:
        print(f"{D:<13} | {results[D]*100:.2f}%")
        
    # Find smallest D within 5% of max accuracy
    max_acc = max(results.values())
    threshold = max_acc - 0.05
    optimal_D = max(d_values)
    
    for D in sorted(d_values):
        if results[D] >= threshold:
            optimal_D = D
            break
            
    print(f"\nMax Accuracy: {max_acc*100:.2f}% (at D={max(d_values, key=results.get)})")
    print(f"Optimal D (within 5%): {optimal_D} (Accuracy: {results[optimal_D]*100:.2f}%)")
    print(f"Compression Factor vs 10000: {10000/optimal_D:.2f}x")

if __name__ == "__main__":
    run_sweep()
