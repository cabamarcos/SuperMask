from keras.layers import Dense
import numpy as np

def verify(model):
    print("Verifying model:")
    for layer in model.layers:
        if isinstance(layer, Dense):
            weights, _ = layer.get_weights()
            total_weights = weights.size
            null_weights = np.sum(weights == 0)
            null_percentage = (null_weights / total_weights) * 100
            print(f"Layer: {layer.name}")
            print(f"Total weights: {total_weights}")
            print(f"Null weights: {null_weights}")
            print(f"Percentage of null weights: {null_percentage:.2f}%")
            print("\n")