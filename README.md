# MIDI to Audio Generator

A deep learning-based music generation system that uses a Transformer model to generate piano music from MIDI files. The model learns patterns from existing MIDI compositions and generates new musical sequences, which can be converted to audio files.

## Features

- **MIDI Processing**: Parse and extract note sequences from MIDI files
- **Transformer-based Generation**: Custom Transformer architecture for learning musical patterns
- **Music Generation**: Generate new piano compositions from seed notes
- **Audio Export**: Convert generated sequences to MIDI and WAV audio files

## Requirements

### Python Dependencies

```bash
pip install pretty_midi midi2audio matplotlib torch miditok miditoolkit tqdm numpy
```

> **Note**: `numpy` is typically installed with `torch`, and `subprocess` and `IPython.display` are part of the Python standard library and Jupyter environment respectively.

### System Dependencies

- **FluidSynth**: Required for MIDI to audio conversion

```bash
# Ubuntu/Debian
apt-get install -y fluidsynth

# macOS
brew install fluidsynth
```

- **SoundFont**: A SoundFont file (e.g., `FluidR3_GM.sf2`) is required for audio synthesis

## Model Architecture

The system uses a custom Transformer model with the following components:

| Component | Description |
|-----------|-------------|
| **Embedding Size** | 256 |
| **Attention Heads** | 4 |
| **Transformer Layers** | 2 |
| **Block Size** | 128 |

### Architecture Components

- **AttentionHead**: Single-head self-attention mechanism with causal masking
- **MultiHeadAttention**: Multi-head attention combining multiple attention heads
- **FeedForward**: Position-wise feed-forward network with ReLU activation
- **TransformerBlock**: Complete transformer block with layer normalization and residual connections
- **MidiTransformer**: Main model with token/position embeddings and language modeling head

## Usage

### 1. Process MIDI Dataset

Update the `midi_folder` variable to point to your MIDI dataset:

```python
midi_folder = "path_to_your_dataset"
process_all_midis(midi_folder)
```

This will create a `notes.txt` file containing tokenized note sequences.

### 2. Train the Model

The model trains on the processed note sequences:

```python
model = MidiTransformer(vocab_size=vocab_size).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

# Set training epochs (adjust based on dataset size and desired quality)
total_epochs = 50  # Start with 50, increase for better results
```

Checkpoints are saved during training to allow resuming.

### 3. Generate Music

Provide seed notes to generate new compositions:

```python
seed_notes = ["D3", "E5", "B3"]
temperature = 1.0  # Higher = more random, Lower = more deterministic
generated_notes = generate_notes(model, input_seq, max_length=100, temperature=temperature)
```

### 4. Export to Audio

Convert generated notes to MIDI and WAV:

```python
# Generate MIDI file
notes_to_midi(generated_notes, output_path="generated.mid")

# Convert to WAV audio
midi_to_wav("generated.mid", "generated.wav", soundfont_path)
```

## How It Works

1. **Data Preparation**: MIDI files are parsed to extract note sequences. Each note is represented by its pitch name (e.g., "C4", "D#5").

2. **Tokenization**: Notes are converted to numerical tokens for model input.

3. **Training**: The Transformer model learns to predict the next note given a sequence of previous notes.

4. **Generation**: Starting from seed notes, the model generates new notes one at a time using temperature-controlled sampling.

5. **Audio Conversion**: Generated note sequences are converted to MIDI format, then synthesized to audio using FluidSynth.

## Configuration

Key parameters that can be adjusted:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BLOCK_SIZE` | 128 | Maximum sequence length for attention |
| `NUM_EMBED` | 256 | Embedding dimension |
| `NUM_HEADS` | 4 | Number of attention heads |
| `NUM_LAYERS` | 2 | Number of transformer blocks |
| `temperature` | 1.0 | Sampling temperature (0.1-2.0 recommended) |
| `max_length` | 100 | Number of notes to generate |

## File Structure

```
MIDItoAudioGenerator/
├── miditoaudio.py              # Main script with all components
├── README.md                   # This file
├── notes.txt                   # Generated during processing (tokenized notes)
├── midi_checkpoint_epoch*.pt   # Training checkpoints (created during training)
├── generated.mid               # Generated MIDI output
└── generated.wav               # Generated audio output
```

## Notes

- This script is designed to run in a Kaggle notebook environment
- GPU acceleration is automatically used when available (`cuda` device)
- Training on larger datasets and more epochs will improve generation quality
- Adjust the `temperature` parameter to control creativity vs. consistency in generation

## License

This project is open source and available under the MIT License.
