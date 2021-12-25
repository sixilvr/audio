import os
if os.name == "nt":
    import winsound
    import matplotlib.pyplot as plt
import time
import numpy as np

from audio.sound import Sound

notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
C0 = 16.351597831287414

def freq(notename):
    octave = int(notename[-1])
    key = notes.index(notename[:-1])
    return C0 * 2. ** ((octave * 12. + key) / 12.)

def transpose(rootnote, newnote):
    return freq(newnote) / freq(rootnote)

def note(freq):
    value = np.log2(freq / C0) * 12.
    return f"{notes[round(value % 12)]}{int(value // 12)}"

def todb(amplitude):
    if np.abs(amplitude) < .000016:
        return -96.
    return 20 * np.log10(np.abs(amplitude))

def toamp(db):
    return 10. ** (db / 20.)

def tosamples(bpm, beat, samplerate = 44100.):
    return int(samplerate * 60. / bpm * bpm)

def fundamental(sound):
    transform = sound.fft()
    return np.argmax(transform) / transform.size * sound.samplerate / 2

def scale(rootnote, major = True):
    seq_major = [0, 2, 4, 5, 7, 9, 11]
    seq_minor = [0, 2, 3, 5, 7, 8, 10]
    start = notes.index(rootnote)
    out = list(map(lambda n: notes[(start + n) % len(notes)], seq_major if major else seq_minor))
    return out

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

def playfile(filename):
    if os.name == "nt":
        winsound.PlaySound(filename, winsound.SND_FILENAME)
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
    return 1 / np.mean(times) * 60
