#todo: Reverb/UI, autotune, better error messages

import os
import warnings
if os.name == "nt":
    import winsound
    import matplotlib.pyplot as plt
    from scipy.fft import rfft, irfft
else:
    from scipy.fftpack import rfft, irfft
import numpy as np
from scipy.io import wavfile
from scipy import signal as sig

tau = np.pi * 2

class Sound:
    def __init__(self, length = 44100, file = None, data = None, samplerate = 44100):
        if file:
            self.read(file)
        elif data is not None:
            self.data = data.astype(np.float32)
            self.samplerate = samplerate
        else:
            self.data = np.zeros(length, dtype = np.float32)
            self.samplerate = samplerate

    def __repr__(self):
        return f"Sound(length = {self.length}, samplerate = {self.samplerate})"

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.data[key]
        elif isinstance(key, float):
            whole, frac = divmod(key, 1)
            whole = int(whole)
            return (1 - frac) * self.data[whole] + frac * self.data[whole + 1]
        else:
            return 0.

    def __setitem__(self, key, value):
        self.data[key] = value

    @property
    def length(self):
        return self.data.size

    @property
    def seconds(self):
        return self.length / self.samplerate

    def copy(self):
        return Sound(length = self.length, data = self.data, samplerate = self.samplerate)

    def sub_sound(self, start_index, end_index):
        return Sound(data = self.data[start_index:end_index])

    def read(self, filename):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.samplerate, self.data = wavfile.read(filename + (".wav" if not filename.endswith(".wav") else ""))
        self.data = self.data.astype(np.float32)
        if self.data.ndim != 1:
            print("Warning: flattening channels for ..." + filename[-20:])
            self.data = np.sum(self.data, 1) / self.data.shape[1]

    def save(self, filename, clip = False):
        if not filename.endswith(".wav"):
            filename += ".wav"
        if clip:
            self.distort(1.)
        wavfile.write(filename, self.samplerate, self.data.astype(np.float32))

    def play(self, sync = True):
        if os.name == "nt":
            self.save("__temp__.wav")
            utils.playfile("__temp__.wav", sync)
            os.remove("__temp__.wav")
        else:
            print("Can only play on Windows")

    def show(self):
        if os.name != "nt":
            print("Can only show on Windows")
            return
        plt.title("Sound")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.ylim(-1.1, 1.1)
        time = self.length / self.samplerate
        time_axis = np.linspace(0, time, self.length)
        plt.plot(time_axis, self.data, linewidth = .75)
        plt.plot([0., time], [0., 0.], "k--", linewidth = .5)
        plt.plot([0., time], [1., 1.], "r--", linewidth = .5)
        plt.plot([0., time], [-1., -1.], "r--", linewidth = .5)
        plt.show()

    def fft(self):
        return rfft(self.data)

    @property
    def fundamental(self):
        transform = np.abs(self.fft())
        return np.argmax(transform) / transform.size * self.samplerate / 2

    def show_fft(self):
        transform = np.abs(self.fft()) / (self.length * 2)
        time_axis = np.linspace(0, 22050, transform.size)
        plt.xlabel("Frequency")
        plt.ylabel("Amplitude")
        #plt.xscale("log")
        plt.plot(time_axis, transform)
        plt.show()

    def ifft(self, transform):
        self.data = irfft(transform)

    @property
    def amplitude(self):
        return np.maximum(self.data.max(), -self.data.min())

    @property
    def rms(self):
        return np.sqrt(np.mean(np.square(self.data)))

    def sine(self, frequency, amplitude = 1.):
        self.data += np.sin(np.linspace(0, frequency * tau * self.length / self.samplerate, self.length)) * amplitude

    def square(self, frequency, amplitude = 1.):
        self.data += sig.square(np.arange(self.length) * tau * frequency / self.samplerate) * amplitude

    def sawtooth(self, frequency, amplitude = 1.):
        self.data += sig.sawtooth(np.arange(self.length) * tau * frequency / self.samplerate + np.pi) * amplitude

    def triangle(self, frequency, amplitude = 1.):
        self.data += sig.sawtooth(np.arange(self.length) * tau * frequency / self.samplerate + np.pi / 2., .5) * amplitude

    def chirp(self, freq_start, freq_end, exponent = 1., amplitude = 1.):
        # y = sin(tau * (c * x^z + f0*x))
        #     where c = (f1-f0) / ((z+1) * t^z))
        c = (freq_end - freq_start) / ((exponent + 1) * self.seconds ** exponent)
        x = np.linspace(0, self.seconds, self.length)
        t = c * x ** (exponent + 1) + freq_start * x
        self.data += np.sin(tau * t) * amplitude

    def noise(self, amplitude = 1.):
        self.data += np.random.default_rng().uniform(-1., 1., self.length) * amplitude

    def resize(self, newsize):
        if newsize > self.length:
            self.data = np.pad(self.data, (0, newsize - self.length))
        else:
            self.data = self.data[:newsize]

    def append(self, sound2):
        self.data = np.append(self.data, sound2.data)

    def mute(self):
        self.data[:] = 0.

    def trim_silence(self, threshold = 0.005):
        start_index = 0
        for i in range(1, self.length):
            if np.abs(self.data[i]) > threshold:
                start_index = i - 1
                break
        end_index = self.length - 1
        for i in range(self.length - 2, 0, -1):
            if np.abs(self.data[i] > threshold):
                end_index = i + 1
                break
        self.data = self.data[start_index:end_index]

    def set_at(self, sound2, offset = 0, multiplier = 1.):
        limit = np.minimum(sound2.length, self.length - offset)
        self.data[offset:offset + limit] = sound2.data[:limit] * multiplier

    def add(self, sound2, offset = 0, multiplier = 1.):
        limit = np.minimum(sound2.length, self.length - offset)
        try:
            self.data[offset:offset + limit] += sound2.data[:limit] * multiplier
        except:
            print(self.length, offset, limit)

    def invert(self):
        self.data = -self.data

    def reverse(self):
        self.data = np.flip(self.data)

    def amplify(self, factor):
        self.data *= factor

    def normalize(self, peak = 1.):
        self.data *= peak / self.amplitude

    def distort(self, threshold = 1.):
        self.data = np.clip(self.data, -threshold, threshold)

    def bit_crush(self, bits = 8):
        factor = np.float32(1 << (bits - 1))
        self.data = np.round(self.data * factor) / factor

    def power(self, exponent = 1.):
        self.data = np.sign(self.data) * np.power(np.abs(self.data), exponent)

    def fade(self, start_index = 0, end_index = None, start_amp = 1., end_amp = 0., exponent = 1.):
        if start_index < 0:
            start_index = 0
        end_index = end_index or self.length
        numsamples = end_index - start_index
        self.data[start_index:end_index] *= np.linspace(start_amp, end_amp, numsamples) ** exponent

    def mavg(self, amount = 2):
        self.data = sig.convolve(self.data, np.repeat(1. / amount, amount))

    def convolve(self, kernel):
        if isinstance(kernel, Sound):
            self.data = sig.convolve(self.data, kernel.data)
        else:
            self.data = sig.convolve(self.data, kernel)

    def stretch(self, factor = 1., in_place = True):
        if not in_place:
            return Sound(data = sig.resample(self.data, int(self.length / factor)), samplerate = self.samplerate)
        self.data = sig.resample(self.data, int(self.length / factor))

    def filter(self, type_, cutoff, order = 2):
        if type_ not in ["lp", "hp", "bp", "bs"]:
            print("Invalid filter type:", type_)
            return
        sos = sig.butter(order, cutoff, type_, output = "sos", fs = self.samplerate)
        self.data = sig.sosfilt(sos, self.data)

    def filter_curve(self, response):
        if isinstance(response, Sound):
            kernel = response.ifft()
        else:
            kernel = Sound()
            kernel.ifft(np.array(response))
        self.convolve(kernel)

    def autotune(self, window_size = 7000):
        out = Sound(self.length)
        sliced = np.array_split(self.data, np.ceil(self.length / window_size))
        for i, part in enumerate(sliced):
            s = Sound(data = part)
            freq = s.fundamental
            newfreq = utils.nearest(freq)
            factor = newfreq / freq
            s.stretch(factor)
            s.fade(end_index = 50, start_amp = 0, end_amp = 1)
            s.fade(start_index = s.length - 50)
            out.set_at(s, window_size * i)
        return out

    def vocode(self, window_size = 7500, harmonics = 3, fade = 50):
        out = Sound(self.length)
        sliced = np.array_split(self.data, np.ceil(self.length / window_size))
        for i, part in enumerate(sliced):
            part_sound = Sound(data = part)
            transform = part_sound.fft()
            values_to_mute = np.argsort(np.abs(transform))[:-harmonics]
            transform[values_to_mute] = 0
            part_sound.ifft(transform)
            part_sound.fade(end_index = fade, start_amp = 0, end_amp = 1)
            part_sound.fade(start_index = part_sound.length - fade)
            out.set_at(part_sound, window_size * i)
        return out

from audio import utils
