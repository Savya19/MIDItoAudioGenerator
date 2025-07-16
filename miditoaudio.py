# Required installations

!pip install pretty_midi midi2audio matplotlib
!apt-get update -qq
!apt-get install -y fluidsynth

!pip install miditok miditoolkit

# Processing dataset

import os
from glob import glob
import pretty_midi
import random


midi_folder = "path_to_your_dataset"


def midi_to_note_sequence(midi_file, min_duration=0.1):
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_file)
        notes = []
        for instrument in midi_data.instruments:
            if instrument.is_drum:
                continue
            for note in instrument.notes:
                if note.end - note.start >= min_duration:
                    notes.append((note.start, note.pitch))
        notes.sort()
        return " ".join(pretty_midi.note_number_to_name(pitch) for _, pitch in notes)
    except Exception as e:
        print(f"Error in {midi_file}: {e}")
        return ""


def process_all_midis(folder, out_txt="notes.txt"):
    midi_files = glob(os.path.join(folder, "**/*.mid"), recursive=True) + \
                 glob(os.path.join(folder, "**/*.midi"), recursive=True)
    print(f"Found {len(midi_files)} MIDI files.")

    random.shuffle(midi_files)
    print(f"Processing all {len(midi_files)} MIDI files...")

    with open(out_txt, "w") as out_file:
        for mf in midi_files:
            notes = midi_to_note_sequence(mf)
            if notes:
                out_file.write(notes + "\n")
process_all_midis(midi_folder)

# Tokenization of the midi files

with open("notes.txt") as f:
    notes = [n.strip() for n in f.read().split() if n.strip()]

unique_notes = sorted(set(notes))
note_to_idx = {note: idx for idx, note in enumerate(unique_notes)}
idx_to_note = {idx: note for note, idx in note_to_idx.items()}
vocab_size = len(note_to_idx)

# Transformer model

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BLOCK_SIZE = 128
NUM_EMBED = 256
NUM_HEADS = 4
NUM_LAYERS = 2

class AttentionHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(NUM_EMBED, head_size, bias=False)
        self.query = nn.Linear(NUM_EMBED, head_size, bias=False)
        self.value = nn.Linear(NUM_EMBED, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(NUM_EMBED, NUM_EMBED)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NUM_EMBED, 4 * NUM_EMBED),
            nn.ReLU(),
            nn.Linear(4 * NUM_EMBED, NUM_EMBED),
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = NUM_EMBED // NUM_HEADS
        self.sa = MultiHeadAttention(NUM_HEADS, head_size)
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(NUM_EMBED)
        self.ln2 = nn.LayerNorm(NUM_EMBED)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class MidiTransformer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, NUM_EMBED)
        self.position_embedding = nn.Embedding(BLOCK_SIZE, NUM_EMBED)
        self.blocks = nn.Sequential(*[TransformerBlock() for _ in range(NUM_LAYERS)])
        self.ln_f = nn.LayerNorm(NUM_EMBED)
        self.lm_head = nn.Linear(NUM_EMBED, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=DEVICE))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm

class NoteDataset(Dataset):
    def __init__(self, filepath, note_to_idx, seq_len=64, verbose=False):
        with open(filepath) as f:
            notes = f.read().split()

        self.note_to_idx = note_to_idx
        self.seq_len = seq_len
        self.verbose = verbose

        raw_notes = notes
        self.tokens = [self.note_to_idx[n] for n in raw_notes if n in self.note_to_idx]
        skipped = len(raw_notes) - len(self.tokens)

        if skipped > 0:
            print(f"Warning: Skipped {skipped} unknown notes.")

        if len(self.tokens) <= self.seq_len + 1:
            raise ValueError(f"Not enough tokens ({len(self.tokens)}) for seq_len {seq_len}. "
                             f"Need at least {self.seq_len + 2} tokens.")

    def __len__(self):
        return len(self.tokens) - self.seq_len - 1

    def __getitem__(self, idx):
        try:
            if idx < 0 or idx + self.seq_len + 1 > len(self.tokens):
                raise IndexError(f"Invalid index: {idx}")
            
            x = torch.tensor(self.tokens[idx:idx+self.seq_len], dtype=torch.long)
            y = torch.tensor(self.tokens[idx+1:idx+self.seq_len+1], dtype=torch.long)
            return x, y

        except Exception as e:
            if self.verbose:
                print(f"Error at idx {idx}: {e}")
                
            return torch.zeros(self.seq_len, dtype=torch.long), torch.zeros(self.seq_len, dtype=torch.long)

# Training the model to generate piano notes

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import os




dataset = NoteDataset("notes.txt", note_to_idx, seq_len=BLOCK_SIZE)
train_len = int(0.9 * len(dataset))
train_set, val_set = random_split(dataset, [train_len, len(dataset) - train_len])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=32, num_workers=0, pin_memory=True)




model = MidiTransformer(vocab_size=vocab_size).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)



checkpoint_path = "/kaggle/input/epoch66/midi_checkpoint_epoch66.pt" 
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']
else:
    start_epoch = 0
    train_losses = []
    val_losses = []




total_epochs = 66
for epoch in range(start_epoch, total_epochs):
    model.train()
    total_train_loss = 0.0
    train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1} [Training]", leave=False)
    
    for x, y in train_loop:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        train_loop.set_postfix(loss=loss.item())

    avg_train_loss = total_train_loss / len(train_loader)

    model.eval()
    total_val_loss = 0.0
    val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1} [Validation]", leave=False)
    
    with torch.no_grad():
        for x, y in val_loop:
            x, y = x.to(DEVICE), y.to(DEVICE)
            _, loss = model(x, y)
            total_val_loss += loss.item()
            val_loop.set_postfix(loss=loss.item())

    avg_val_loss = total_val_loss / len(val_loader)
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses
    }, f"/kaggle/working/midi_checkpoint_epoch{epoch+1}.pt")


plt.plot(range(len(train_losses)), train_losses, label="Train")
plt.plot(range(len(val_losses)), val_losses, label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Curve")
plt.legend()
plt.xticks(range(0, len(train_losses), 5))  
plt.savefig("/kaggle/working/training_curv.png")
plt.show()

torch.save(model.state_dict(), "/kaggle/working/midi_transformer.pt")

# Generating piano notes to continue the prompt

import torch
import numpy as np


model = MidiTransformer(vocab_size=vocab_size).to(DEVICE)
model.load_state_dict(torch.load("/kaggle/input/midi65/midi_transformer (5).pt", map_location=DEVICE))
model.eval()


seed_notes = ["D3", "E5", "B3"] 
seed_idx = [note_to_idx[note] for note in seed_notes]


input_seq = torch.tensor(seed_idx, dtype=torch.long).unsqueeze(0).to(DEVICE)


def generate_notes(model, input_seq, max_length=100, temperature=1.0):
    model.eval()
    generated = input_seq.clone()
    
    for _ in range(max_length):
        if generated.size(1) > BLOCK_SIZE:
            input_chunk = generated[:, -BLOCK_SIZE:]
        else:
            input_chunk = generated

        logits, _ = model(input_chunk)
        next_token_logits = logits[:, -1, :]

        
        next_token_logits = next_token_logits / temperature

        probabilities = torch.softmax(next_token_logits, dim=-1)

    
        next_token = torch.multinomial(probabilities, num_samples=1)
        generated = torch.cat((generated, next_token), dim=1)

    return generated.squeeze().tolist()


temperature = 1.0 
generated_indices = generate_notes(model, input_seq, max_length=100, temperature=temperature)


generated_notes = [idx_to_note[idx] for idx in generated_indices]


print("Generated Notes:")
print(generated_notes)

# Converting the tokens (piano notes) to midi file and saving

import pretty_midi

print("Running MIDI conversion...") 

def notes_to_midi(note_names, output_path="generated.mid"):
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0) 

    start_time = 0
    duration = 0.5  

    for note_name in note_names:
        try:
            pitch = pretty_midi.note_name_to_number(note_name)
            note = pretty_midi.Note(velocity=100, pitch=pitch, start=start_time, end=start_time + duration)
            instrument.notes.append(note)
            start_time += duration
        except:
            print(f"Skipping invalid note: {note_name}")

    midi.instruments.append(instrument)
    midi.write(output_path)
    print(f"MIDI file saved as: {output_path}")

notes_to_midi(generated_notes, output_path="generated.mid")

# Converting the midi file to audio

import subprocess
from IPython.display import FileLink


def midi_to_wav(midi_path="generated.mid", wav_path="generated.wav", soundfont_path="/kaggle/input/sf2soundfont/FluidR3_GM.sf2"):
    result = subprocess.run([
        "fluidsynth", "-ni", soundfont_path, midi_path, "-F", wav_path, "-r", "44100"
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print("Fluidsynth failed with error:")
        print(result.stderr)
    else:
        print("Conversion successful! WAV saved to:", wav_path)

midi_path = "generated.mid"  
wav_path = "generated.wav"
soundfont_path = "/kaggle/input/sf2soundfont/FluidR3_GM.sf2"  

midi_to_wav(midi_path, wav_path, soundfont_path)

print("Click below to download the WAV file:")
FileLink(wav_path)


