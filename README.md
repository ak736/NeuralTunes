# Neural Tunes: AI Music Generator 🎵

## Overview
Neural Tunes is an advanced machine learning project that generates original music sequences using Long Short-Term Memory (LSTM) neural networks. The system learns patterns from MIDI files to create new musical compositions, demonstrating the potential of AI in creative applications.

## 🛠️ Technical Details

### Architecture
- **Model Type**: Multi-layer LSTM Neural Network
- **Layer Configuration**:
  ```python
  model = Sequential([
      LSTM(256, return_sequences=True),
      Dropout(0.3),
      LSTM(256),
      Dropout(0.3),
      Dense(128, activation='relu'),
      Dense(output_size, activation='softmax')
  ])
  ```

### Performance Metrics
- Training Accuracy: ~10%
- Validation Loss: 5.67
- Total Parameters: 822,400
- Training Data: 14,495 notes processed
- Unique Notes: 174

### 🔑 Key Features
1. **Preprocessing Pipeline**
   - MIDI file parsing using Music21
   - Note sequence generation
   - One-hot encoding for musical notes
   - Data normalization

2. **Model Architecture**
   - Sequential note prediction
   - Temperature-based sampling
   - Dropout layers for regularization
   - Adam optimizer with categorical crossentropy loss

3. **Generation Capabilities**
   - Multiple music styles (balanced, creative, structured)
   - Variable sequence lengths
   - Adjustable creativity parameters

## 🚀 Installation and Usage

### Prerequisites
```bash
# For M1/M2 Macs
pip install tensorflow-macos
pip install tensorflow-metal

# Required packages
pip install music21
pip install numpy
pip install matplotlib
```

### Project Structure
```
project/
├── data/
│   ├── MIDI files
│   └── mappings/
├── models/
│   └── trained_model.keras
├── output/
│   └── generated music files
└── scripts/
    ├── preprocessing.py
    ├── model.py
    ├── training.py
    └── generate.py
```

### Running the Project
```bash
# 1. Preprocess data
python preprocessing.py

# 2. Train model
python training.py

# 3. Generate music
python generate.py
```

## 📊 Model Performance and Improvements

### Current Limitations
- Validation loss indicates room for improvement
- Limited musical structure learning
- Basic rhythm patterns

### Potential Improvements
1. **Architecture Enhancements**
   ```python
   # Example of enhanced architecture
   model = Sequential([
       Bidirectional(LSTM(512, return_sequences=True)),
       Attention(),
       LSTM(512),
       Dense(256, activation='relu'),
       Dense(output_size, activation='softmax')
   ])
   ```

2. **Data Augmentation**
   - Increase training data size
   - Include diverse musical genres
   - Add rhythm and tempo variations

3. **Training Optimization**
   ```python
   model.compile(
       optimizer=Adam(learning_rate=0.001),
       loss='categorical_crossentropy',
       metrics=['accuracy']
   )
   ```

## 🎯 Real-World Applications

1. **Music Industry**
   - Background music generation
   - Inspiration for composers
   - Automated jingle creation

2. **Entertainment**
   - Video game soundtrack generation
   - Interactive music systems
   - Dynamic content creation

3. **Education**
   - Music theory demonstration
   - Composition assistance
   - Pattern recognition training

## 💡 Technical Challenges Overcome
1. Memory optimization for M2 MacBook Air
   ```python
   # Memory optimization
   physical_devices = tf.config.list_physical_devices('GPU')
   for device in physical_devices:
       tf.config.experimental.set_memory_growth(device, True)
   ```
2. MIDI file processing and conversion
3. Sequence generation optimization
4. Model architecture balancing

## 🔮 Future Enhancements
1. Implement transformer architecture
2. Add genre-specific training
3. Develop real-time generation capabilities
4. Integrate with audio synthesis

## Output Examples
Generated music samples can be found in the `output/` directory:
- `generated_balanced.mid`: Standard generation
- `generated_creative.mid`: More experimental
- `generated_structured.mid`: More predictable

## Model Training Results
![Training History](models/training_history.png)



## 🙏 Acknowledgments
- TensorFlow and Keras documentation
- Music21 library documentation
- Deep learning music generation research papers

## 👤 Author
Your Name
- GitHub: [@ak736](https://github.com/ak736)
- LinkedIn: [Aniket kumar](https://www.linkedin.com/in/aniket736)

## 🤝 Contributing
Contributions, issues, and feature requests are welcome!

## Show your support
Give a ⭐️ if this project helped you!