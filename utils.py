import os
if os.name == "nt":
    import winsound
    import matplotlib.pyplot as plt
from scipy.fft import rfft
import time
import re
import numpy as np

from .sound import Sound

NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MAJOR_SCALE = np.array([0, 2, 4, 5, 7, 9, 11, 12])
MINOR_SCALE = np.array([0, 2, 3, 5, 7, 8, 11, 12])

note_regex = re.compile(r"[A-G]#?\d")

def frequency_to_midi(frequency):
    return int(round(np.log2(frequency / 440) * 12 + 69))

def midi_to_frequency(midi):
    return 440 * 2 ** ((midi - 69) / 12)

def note_to_midi(note):
    if not re.match(note_regex, note):
        raise ValueError(f"Invalid note name: {note}")
    octave = int(note[-1])
    key = NOTES.index(note[:-1])
    return 12 * (octave + 1) + key

def midi_to_note(midi):
    octave, key = divmod(midi, 12)
    return NOTES[key] + str(octave - 1)

def note_to_frequency(note):
    return midi_to_frequency(note_to_midi(note))

def frequency_to_note(frequency):
    return midi_to_note(frequency_to_midi(frequency))

def transpose_factor(root_note, new_note):
    return note_to_frequency(new_note) / note_to_frequency(root_note)

def round_frequency(frequency):
    return midi_to_frequency(frequency_to_midi(frequency))

def amplitude_to_db(amplitude):
    if np.abs(amplitude) < 0.000016:
        return -96.0
    return 20 * np.log10(np.abs(amplitude))

def db_to_amplitude(db):
    return 10 ** (db / 20)

def beats_to_samples(bpm, beats, samplerate = 44100):
    return int(round(60 * samplerate * beats / bpm))

def samples_to_beats(bpm, samples, samplerate = 44100):
    return int(round(samples * bpm / samplerate / 60))

def scale(rootnote, quality = "major"):
    if quality not in ["major", "minor"]:
        raise ValueError(f"Expected scale type \"major\" or \"minor\", got {quality}")
    seq_major = [0, 2, 4, 5, 7, 9, 11]
    seq_minor = [0, 2, 3, 5, 7, 8, 10]
    start = NOTES.index(rootnote)
    out = map(lambda n: NOTES[(start + n) % len(NOTES)], seq_major if quality == "major" else seq_minor)
    return list(out)

def chord(scale, order = 1, amount = 3):
    out = [scale[(order - 1) % len(scale)]]
    for i in range(1, amount):
        out.append(scale[(order + i * 2) % len(scale)])
    return out

def plot(*data):
    for sound in data:
        if isinstance(sound, Sound):
            plt.plot(sound.data)
        else:
            plt.plot(sound)
    plt.show()

def play_file(filename, sync = True):
    if os.name == "nt":
        winsound.PlaySound(filename, winsound.SND_FILENAME | (0 if sync else winsound.SND_ASYNC))
    else:
        print("Warning: utils.playfile only works on Windows")

def tempo_tapper(limit = 10, amount = 8):
    times = np.zeros(amount)
    print(f"Press enter for each beat, {limit} times")
    last_time = time.monotonic()
    for _ in range(limit):
        input()
        times = np.roll(times, -1)
        times[-1] = time.monotonic() - last_time
        last_time = time.monotonic()
    return 60 / np.mean(times)

def split_zero_cross(sound, minsize = 100, threshold = 0.015):
    step0 = 0
    i = minsize
    length = sound.length
    out = []
    while i < length:
        if np.abs(sound.data[i]) < threshold:
            out.append(np.copy(sound.data[step0:i]))
            step0 = i + 1
            i = step0 + minsize
        else:
            i += 1
    out.append(np.copy(sound.data[step0:]))
    return out

def extract_frequencies(sound, window_size = 5000, tolerance = 10.):
    splits = np.array_split(sound.data, sound.length // window_size)
    freqs = np.zeros(len(splits))
    for i, split in enumerate(splits):
        transform = rfft(split)
        freqs[i] = np.argmax(transform) / len(transform) * sound.samplerate
    return freqs

def tone(freq = 440., numsamples = 22050, amplitude = 0.75):
    out = Sound(numsamples)
    out.sine(freq, amplitude)
    return out

def play_tone(freq = 440, numsamples = 22050, amplitude = 0.75):
    out = tone(freq, numsamples, amplitude)
    out.play()

def play_data(data):
    out = np.Sound(data = data)
    out.play()

