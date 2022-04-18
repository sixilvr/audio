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

    def copy(self):
        return Sound(self.length, self.samplerate, data = self.data)

    def read(self, filename):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.samplerate, self.data = wavfile.read(filename + (".wav" if not filename.endswith(".wav") else ""))
        self.data = self.data.astype(np.float32)
        if self.data.ndim != 1:
            print("Warning: flattening channels for", filename)
            self.data = np.sum(self.data, 1)

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
        for i, sample in np.ndenumerate(self.data):
            if sample > threshold:
                self.data[i] = threshold
            elif sample < -threshold:
                self.data[i] = -threshold

    def bitcrush(self, bits = 8):
        factor = np.float32(1 << (bits - 1))
        self.data = np.round(self.data * factor) / factor

    def power(self, exponent = 1.):
        for i, sample in np.ndenumerate(self.data):
            if sample >= 0.:
                self.data[i] = np.power(sample, exponent)
            else:
                self.data[i] = -np.power(-sample, exponent)

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
            return Sound(samplerate = self.samplerate, data = sig.resample(self.data, int(self.length / factor)))
        self.data = sig.resample(self.data, int(self.length / factor))

    def filter(self, cutoff, type_ = "lp", order = 2):
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
            kernel.ifft(response)
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

    def vocode(self, window_size = 7000, fade = 50):
        #https://stackoverflow.com/a/23734295
        out = Sound(self.length)
        sliced = np.array_split(self.data, np.ceil(self.length / window_size))
        for i, part in enumerate(sliced):
            s = Sound(data = part)
            t = s.fft()
            m = np.argmax(t)
            t[:m] = 0
            t[m + 1:] = 0
            s.ifft(t)
            s.fade(end_index = fade, start_amp = 0, end_amp = 1)
            s.fade(start_index = s.length - fade)
            out.set_at(s, window_size * i)
        return out

from audio import utils
