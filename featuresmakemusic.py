import cv2
import numpy as np
import pretty_midi
from keras.applications import VGG16
from keras.models import Model

# Load the pre-trained VGG16 model
base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)  # Use the second last layer

# Function to extract features from an image
def extract_image_features(image_path):
    # Load the image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Resize to VGG input size
    img = img.astype('float32') / 255  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Extract features
    features = model.predict(img)
    return features.flatten()

# Function to generate MIDI from features
def generate_midi_from_features(features, midi_file_path):
    # Create a PrettyMIDI object
    midi = pretty_midi.PrettyMIDI()
    
    # Create an instrument
    instrument = pretty_midi.Instrument(program=0)
    
    # Map features to MIDI notes (simple mapping)
    for i, feature in enumerate(features):
        note_number = int(np.clip(feature * 127, 0, 127))  # Map to MIDI range
        start_time = i * 0.5  # Each note lasts for 0.5 seconds
        end_time = start_time + 0.5
        note = pretty_midi.Note(velocity=100, pitch=note_number, start=start_time, end=end_time)
        instrument.notes.append(note)
    
    # Add instrument to MIDI
    midi.instruments.append(instrument)
    
    # Write to file
    midi.write(midi_file_path)

# Example usage
if __name__ == "__main__":
    image_path = "path/to/your/image.jpg"  # Replace with your image path
    midi_file_path = "output.mid"
    
    # Extract features and generate MIDI
    features = extract_image_features(image_path)
    generate_midi_from_features(features, midi_file_path)
    print(f"MIDI file generated: {midi_file_path}")
