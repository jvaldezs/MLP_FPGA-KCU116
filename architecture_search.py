import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

def load_data(filepath):
    print(f"Loading dataset from {filepath}...")
    # Assuming CSV format: Label, Pixel1, Pixel2, ..., Pixel300
    df = pd.read_csv(filepath, header=None)
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values
    
    # Normalize features if necessary (assuming pixel values 0-255)
    if X.max() > 1.0:
        X = X / 255.0
        
    return X, y

def train_and_evaluate(X_train, X_test, y_train, y_test, hidden_layer_sizes):
    print(f"\n=========================================")
    print(f"Training Architecture: {hidden_layer_sizes}")
    print(f"=========================================")
    
    # Configure an MLP similar to our FPGA constraint (1 hidden layer, relu activation)
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes, 
        activation='relu', 
        solver='adam', 
        max_iter=50,      # Keep iteration count low for quick search
        random_state=42,  # Fixed seed for fair comparison
        verbose=False
    )
    
    start_time = time.time()
    mlp.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Evaluate
    predictions = mlp.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"-> Training Time: {train_time:.2f} seconds")
    print(f"-> Architecture Accuracy: {accuracy * 100:.2f}%")
    return accuracy

def main():
    # 1. Load the same dataset Vitis HLS is using
    dataset_path = "E:/01_A_PROJ/HLS_Export/mnist_test.csv"
    
    try:
        X, y = load_data(dataset_path)
    except FileNotFoundError:
        print(f"Error: Could not find dataset at {dataset_path}")
        return

    # In a real scenario we'd use a separate training set. 
    # Since we only have the test set readily available here, we'll split it 80/20
    # just to demonstrate the loss curves between architectural sizes.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nTotal Dataset Samples: {len(y)}")
    print(f"Training Samples: {len(y_train)} | Test Samples: {len(y_test)}")

    # 2. Define the architectures we want to test
    # (Number of Neurons in Hidden Layer 1)
    architectures_to_test = [
        (10000,),  # Original planned size (Too big for FPGA)
        (6000,),   # Optimized size (Fits FPGA memory pragmas perfectly)
        (3000,)    # Aggressively small (Just to see the accuracy drop)
    ]
    
    results = {}
    
    # 3. Train and compare
    for arch in architectures_to_test:
        acc = train_and_evaluate(X_train, X_test, y_train, y_test, arch)
        results[f"{arch[0]} Neurons"] = acc
        
    # 4. Print final comparison report
    print("\n\n=========================================")
    print("FINAL ACCURACY DEGRADATION REPORT")
    print("=========================================")
    
    baseline = results["10000 Neurons"]
    
    for arch_name, acc in results.items():
        drop = (baseline - acc) * 100
        print(f"{arch_name}: \t {acc * 100:.2f}% \t(Loss vs Baseline: {drop:+.2f}%)")
        
if __name__ == "__main__":
    main()
