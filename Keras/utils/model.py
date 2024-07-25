from keras.models import Sequential
from keras.layers import Dense, Flatten

# Definir la función para crear el modelo
def create_model():
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model