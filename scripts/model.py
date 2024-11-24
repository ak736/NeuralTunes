import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os

def build_model(input_shape, output_size):
    """
    Build an LSTM model for music generation

    Args:
        input_shape: Tuple of (sequence_length, features)
        output_size: Number of possible notes/chords to predict

    Returns:
        Compiled Keras Sequential model
    """
    model = Sequential([
        LSTM(256, input_shape=input_shape, return_sequences=True),
        Dropout(0.3),
        LSTM(256),
        Dropout(0.3),
        Dense(output_size, activation='softmax')
    ])

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    return model

def save_model(model, path='models/trained_model.h5'):
    """Save the trained model"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)

if __name__ == "__main__":
    # Example usage
    sequence_length = 100
    n_features = 1
    output_size = 128  # Number of possible notes
    
    model = build_model(
        input_shape=(sequence_length, n_features),
        output_size=output_size
    )
    print(model.summary())