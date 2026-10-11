import cv2
import numpy as np
from keras.applications import VGG16
from keras.models import Model
import magenta
from magenta.models.melody_rnn import melody_rnn_sequence_generator
from magenta.music import musicxml_reader, midi_io, note_sequence


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


def generate_music_from_features(features):
    # Convert features to a format suitable for music generation
    melody = note_sequence.NoteSequence()
    
    # Map features to musical properties
    for i, feature in enumerate(features[:32]):  # Limit to 32 features for melody
        note = note_sequence.Note()
        note.pitch = int(np.clip(feature * 127, 0, 127))  # Map to MIDI range
        note.start_time = i * 0.5  # Each note lasts for 0.5 seconds
        note.end_time = note.start_time + 0.5
        note.velocity = 80  # Fixed velocity
        melody.notes.append(note)
    
    melody.total_time = len(melody.notes) * 0.5
    
    # Generate music using Magenta's Melody RNN
    # This assumes you have a trained Melody RNN model
    generator = melody_rnn_sequence_generator.get_generator("basic_rnn")
    
    generated_sequence = generator.generate(melody, num_steps=128)  # Generate 128 steps
    return generated_sequence

# Save the generated music to a MIDI file
def save_to_midi(sequence, midi_file_path):
    midi_io.note_sequence_to_midi_file(sequence, midi_file_path)

# Example usage
if __name__ == "__main__":
    image_path = "path/to/your/image.jpg"  # Replace with your image path
    midi_file_path = "output.mid"
    
    # Extract features
    features = extract_image_features(image_path)
    
    # Generate music
    generated_sequence = generate_music_from_features(features)
    
    # Save to MIDI
    save_to_midi(generated_sequence, midi_file_path)
    print(f"MIDI file generated: {midi_file_path}")
