import os
if os.name == "nt":
    import winsound
    import matplotlib.pyplot as plt
from scipy.fft import rfft
import time
import numpy as np

from audio.sound import Sound

notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
twelvertwo = 2 ** (1 / 12)
C0 = 440 * twelvertwo ** -57

def freq(notename):
    octave = int(notename[-1])
    key = notes.index(notename[:-1])
    return C0 * twelvertwo ** (octave * 12. + key)

def nearest(frequency):
    key = round(np.log2(frequency / C0) * 12)
    return C0 * twelvertwo ** key

def transpose(rootnote, newnote):
    return freq(newnote) / freq(rootnote)

def note(frequency):
    value = round(np.log2(frequency / C0) * 12.)
    return f"{notes[value % 12]}{int(value // 12)}"

def todb(amplitude):
    if np.abs(amplitude) < .000016:
        return -96.
    return 20 * np.log10(np.abs(amplitude))

def toamp(db):
    return 10. ** (db / 20.)

def tosamples(bpm, beat, samplerate = 44100.):
    return int(samplerate * 60. / bpm * bpm)

def scale(rootnote, type = "major"):
    seq_major = [0, 2, 4, 5, 7, 9, 11]
    seq_minor = [0, 2, 3, 5, 7, 8, 10]
    start = notes.index(rootnote)
    out = map(lambda n: notes[(start + n) % len(notes)], seq_major if type == "major" else seq_minor)
    return list(out)

def chord(scale, order = 1, amount = 3):
    out = [scale[(order - 1) % len(scale)]]
    for i in range(amount - 1):
        out.append(scale[(order + 1 + i * 2) % len(scale)])
    return out

def plot(*data):
    for sound in data:
        if isinstance(sound, Sound):
            plt.plot(sound.data)
        else:
            plt.plot(sound)
    plt.show()

def playfile(filename, sync = True):
    if os.name == "nt":
        winsound.PlaySound(filename, winsound.SND_FILENAME | (0 if sync else winsound.SND_ASYNC))
    else:
        print("Can only play on Windows")

def tempotapper(limit = 10, amount = 8):
    times = np.zeros(amount)
    print(f"Press enter for each beat, {limit} times")
    last_time = time.monotonic()
    for i in range(limit):
        input()
        times = np.roll(times, -1)
        times[-1] = time.monotonic() - last_time
        last_time = time.monotonic()
    return 60. / np.mean(times)

def splitzerocross(sound, minsize = 100, threshold = 0.015):
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

def extractfrequencies(sound, window_size = 5000, tolerance = 10.):
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

def playtone(freq = 440, numsamples = 22050, amplitude = 0.75):
    out = tone(freq, numsamples, amplitude)
    out.play()

