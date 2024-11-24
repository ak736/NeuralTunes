import os
import json
import numpy as np
from music21 import converter, note, chord
from keras.utils import to_categorical

def preprocess_midi_files(data_path='../data', sequence_length=100):
    """
    Process MIDI files and prepare sequences for training
    """
    notes = []
    print("Reading MIDI files...")
    
    # Get list of MIDI files
    midi_files = [f for f in os.listdir(data_path) if f.endswith('.mid')]
    
    # Process each file
    for file in midi_files:
        try:
            midi = converter.parse(os.path.join(data_path, file))
            notes_to_parse = midi.flat.notes
            
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))
                    
        except Exception as e:
            print(f"\nError processing {file}: {str(e)}")
            continue
    
    print(f"\nTotal notes/chords extracted: {len(notes)}")
    
    # Create mapping dictionaries
    pitchnames = sorted(set(notes))
    print(f"Unique notes/chords found: {len(pitchnames)}")
    
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
    
    # Save mappings
    os.makedirs('../data/mappings', exist_ok=True)
    with open('../data/mappings/note_to_int.json', 'w') as f:
        json.dump(note_to_int, f)
    with open('../data/mappings/int_to_note.json', 'w') as f:
        json.dump(int_to_note, f)
    
    # Prepare sequences
    print("Creating sequences...")
    network_input = []
    network_output = []
    
    for i in range(0, len(notes) - sequence_length):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])
    
    n_patterns = len(network_input)
    n_vocab = len(pitchnames)
    
    # Reshape and normalize input
    X = np.reshape(network_input, (n_patterns, sequence_length, 1))
    X = X / float(n_vocab)
    
    # One-hot encode the output
    y = to_categorical(network_output, num_classes=n_vocab)
    
    # Save preprocessed data
    os.makedirs('../data/processed', exist_ok=True)
    np.save('../data/processed/network_input.npy', X)
    np.save('../data/processed/network_output.npy', y)
    
    print("\nPreprocessing complete!")
    print(f"Training sequences created: {n_patterns}")
    print(f"Vocabulary size: {n_vocab}")
    
    return X, y, n_vocab

if __name__ == "__main__":
    preprocess_midi_files()