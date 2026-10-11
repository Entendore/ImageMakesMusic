import cv2
import numpy as np
from midiutil import MIDIFile

# Function to extract the dominant color from an image
def extract_dominant_color(image_path):
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize the image to reduce processing time
    image = cv2.resize(image, (100, 100))
    
    # Reshape the image to a list of pixels
    pixels = image.reshape(-1, 3)
    
    # Calculate the mean color
    mean_color = np.mean(pixels, axis=0)
    return tuple(mean_color.astype(int))

# Function to map color to MIDI notes
def color_to_midi_color(color):
    r, g, b = color
    
    # Simple mapping of RGB values to MIDI note numbers (60-72 for C4 to B4)
    note_r = int(np.clip(r / 255 * 12 + 60, 60, 72))
    note_g = int(np.clip(g / 255 * 12 + 60, 60, 72))
    note_b = int(np.clip(b / 255 * 12 + 60, 60, 72))
    
    return [note_r, note_g, note_b]

# Function to generate MIDI file from colors
def generate_midi_from_colors(image_path, midi_file_path):
    # Extract dominant color
    dominant_color = extract_dominant_color(image_path)
    
    # Map the dominant color to MIDI notes
    midi_notes = color_to_midi_color(dominant_color)
    
    # Create a MIDI file
    midi = MIDIFile(1)
    midi.addTrackName(0, 0, "Color Music")
    midi.addTempo(0, 0, 120)  # Set tempo to 120 BPM
    
    # Add notes based on the color
    for i, note in enumerate(midi_notes):
        midi.addNote(0, 0, note, i * 1, 1, 100)  # Add note (track, channel, pitch, time, duration, volume)
    
    # Save the MIDI file
    with open(midi_file_path, "wb") as midi_file:
        midi.writeFile(midi_file)

# Example usage
if __name__ == "__main__":
    image_path = "path/to/your/image.jpg"  # Replace with your image path
    midi_file_path = "output.mid"
    
    generate_midi_from_colors(image_path, midi_file_path)
    print(f"MIDI file generated: {midi_file_path}")
