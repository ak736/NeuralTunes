import numpy as np
from music21 import note, chord, stream, instrument, tempo
from tensorflow import keras
import json
import os

def load_mapping(mapping_path='../data/mappings/int_to_note.json'):
    """Load the integer to note mapping from a JSON file"""
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r') as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"Mapping file not found at {mapping_path}")

def generate_music(model_path='../models/trained_model.keras',  # Updated to .keras
                  sequence_length=100,
                  mapping_path='../data/mappings/int_to_note.json',
                  output_name='generated',
                  num_notes=400,       # Number of notes to generate
                  temperature=1.0):    # Control randomness (higher = more random)
    """
    Generate music using the trained model
    """
    print("Loading model and mappings...")
    model = keras.models.load_model(model_path)
    int_to_note = load_mapping(mapping_path)
    
    # Load some seed data for initial sequence
    network_input = np.load('../data/processed/network_input.npy')
    
    # Generate starting sequence
    print("Generating music...")
    start = np.random.randint(0, len(network_input) - 1)
    pattern = network_input[start]
    output_notes = []

    # Generate notes
    for i in range(num_notes):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction = model.predict(prediction_input, verbose=0)
        
        # Apply temperature scaling
        prediction = np.log(prediction) / temperature
        exp_preds = np.exp(prediction)
        prediction = exp_preds / np.sum(exp_preds)
        
        # Add some randomness to note selection
        if np.random.random() < 0.1:  # 10% chance of choosing second best note
            indices = np.argsort(prediction[0])[-2:]
            index = indices[np.random.randint(0, len(indices))]
        else:
            index = np.argmax(prediction)
            
        result = int_to_note[str(index)]
        output_notes.append(result)
        
        # Update pattern
        pattern = np.append(pattern[1:], [[index/len(int_to_note)]], axis=0)

    print("Converting to MIDI...")
    # Create the music stream
    midi_stream = stream.Stream()
    
    # Add tempo marking (adjust as needed)
    midi_stream.append(tempo.MetronomeMark(number=90))
    
    # Add piano instrument
    piano = instrument.Piano()
    midi_stream.append(piano)
    
    # Convert to MIDI
    offset = 0
    for pattern in output_notes:
        try:
            if '.' in pattern:
                # Handle chord
                notes_in_chord = pattern.split('.')
                chord_notes = []
                for current_note in notes_in_chord:
                    new_note = note.Note(int(current_note))
                    chord_notes.append(new_note)
                new_chord = chord.Chord(chord_notes)
                new_chord.offset = offset
                midi_stream.append(new_chord)
            else:
                # Handle single note
                new_note = note.Note(pattern)
                new_note.offset = offset
                midi_stream.append(new_note)
                
            # Add slight variation to note duration for more natural sound
            offset += 0.5 + np.random.uniform(-0.1, 0.1)
            
        except Exception as e:
            print(f"Error processing note {pattern}: {str(e)}")
            continue

    # Create output directory if it doesn't exist
    os.makedirs('../output', exist_ok=True)
    
    # Save the MIDI file
    output_path = f'../output/{output_name}.mid'
    midi_stream.write('midi', fp=output_path)
    print(f"Music saved to {output_path}")
    
    return midi_stream

if __name__ == "__main__":
    try:
        # Generate multiple versions with different settings
        print("Generating first version (balanced)...")
        generate_music(temperature=1.0, output_name='generated_balanced')
        
        print("\nGenerating second version (more creative)...")
        generate_music(temperature=1.2, output_name='generated_creative')
        
        print("\nGenerating third version (more structured)...")
        generate_music(temperature=0.8, output_name='generated_structured')
        
        print("\nMusic generation completed successfully!")
    except Exception as e:
        print(f"Error during music generation: {str(e)}")
        import traceback
        traceback.print_exc()