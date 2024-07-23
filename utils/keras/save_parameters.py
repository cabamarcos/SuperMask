from keras.layers import Dense
import numpy as np

def save_model_parameters_to_file(model, model_name, filename):
    with open(filename, 'w') as f:
        f.write(f"Parameters of model: {model_name}\n")
        for layer in model.layers:
            if isinstance(layer, Dense):
                weights, biases = layer.get_weights()
                f.write(f"Layer: {layer.name}\n")
                f.write("Weights:\n")
                np.savetxt(f, weights, fmt='%.4f')  # Guarda los pesos en el archivo
                f.write("Biases:\n")
                np.savetxt(f, biases, fmt='%.4f')  # Guarda los sesgos en el archivo
                f.write("\n")
