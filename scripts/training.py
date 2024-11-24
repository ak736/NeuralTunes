import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# Configure TensorFlow for memory efficiency
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

def train_model(input_path='../data/processed/network_input.npy',
                output_path='../data/processed/network_output.npy',
                model_save_path='../models/trained_model.keras'):
    """
    Memory and CPU efficient training for M2 MacBook Air
    """
    # Load preprocessed data
    print("Loading training data...")
    X = np.load(input_path)
    y = np.load(output_path)
    
    # Get dimensions for model
    sequence_length = X.shape[1]
    n_features = 1
    output_size = y.shape[1]
    
    print(f"Training data shape: {X.shape}")
    print(f"Output size (unique notes): {output_size}")
    
    # Ensure output directory exists
    os.makedirs('../models', exist_ok=True)
    
    # Setup callbacks with more aggressive early stopping
    callbacks = [
        ModelCheckpoint(
            filepath='../models/best_model.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,  # Reduced patience
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,  # Reduced patience
            min_lr=0.0001,
            verbose=1
        ),
        # Add cooling breaks
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: plt.close('all')  # Clear memory
        )
    ]
    
    # Build smaller model
    print("Building model...")
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(sequence_length, n_features)),
        # Reduced size of LSTM layers
        tf.keras.layers.LSTM(256, return_sequences=True),  # Reduced from 512
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(256),  # Removed one LSTM layer
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),  # Reduced from 256
        tf.keras.layers.Dense(output_size, activation='softmax')
    ])
    
    # Compile with mixed precision
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    
    # Print model summary
    model.summary()
    
    # Train with smaller batches and fewer epochs
    print("\nStarting training...")
    history = model.fit(
        X, 
        y,
        batch_size=32,          # Reduced from 64
        epochs=30,              # Reduced from 100
        validation_split=0.2,
        verbose=1,
        callbacks=callbacks,
        shuffle=True
    )
    
    # Plot training history
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('../models/training_history.png')
    plt.close()  # Clear memory
    print("\nTraining history saved as 'training_history.png'")
    
    # Save final model
    print("Saving model...")
    model.save(model_save_path)
    
    return history

if __name__ == "__main__":
    try:
        # Optional: Add cooling break before training
        input("Press Enter to start training (make sure your laptop is well-ventilated)...")
        
        history = train_model()
        print("Training complete!")
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()