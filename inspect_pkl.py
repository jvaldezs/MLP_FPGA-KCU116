import joblib
import numpy as np
import sys

PKL_PATH = r"c:\Users\jvald\HLS_Export\Weights\TPU_CODE\10000test.pkl"

try:
    data = joblib.load(PKL_PATH)
    print(f"Type: {type(data)}")
    if isinstance(data, (tuple, list)):
        print(f"Length: {len(data)}")
        for i, item in enumerate(data):
            if hasattr(item, 'shape'):
                print(f"Item {i} shape: {item.shape}")
            else:
                print(f"Item {i} type: {type(item)}")
    elif hasattr(data, 'shape'):
        print(f"Shape: {data.shape}")
    else:
        print("Unknown structure")
except Exception as e:
    print(f"Error: {e}")
