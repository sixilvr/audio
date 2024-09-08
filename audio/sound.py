#todo: autotune, save as mp3, arrangement notation pattern, cepstrum for fundamental

import os
import warnings
if os.name == "nt":
    import matplotlib.pyplot as plt
from scipy.fft import rfft, irfft
import numpy as np
from scipy.io import wavfile
from scipy import signal as sig

tau = np.pi * 2

class Sound:

    def __init__(self, length = 44100, file = None, data = None, samplerate = 44100):
        if file:
            self.read(file)
        elif data is not None:
            self.data = np.array(data, dtype = "float32")
            self.samplerate = samplerate
        else:
            self.data = np.zeros(int(length), dtype = np.float32)
            self.samplerate = samplerate

    def __repr__(self):
        return f"audio.Sound(length = {len(self)}, samplerate = {self.samplerate})"

    def __getitem__(self, key):
        if isinstance(key, int):
            if key < 0:
                return 0.
            else:
                return self.data[key]
        elif isinstance(key, float):
            whole, frac = divmod(key, 1)
            whole = int(whole)
            return (1 - frac) * self.data[whole] + frac * self.data[whole + 1]
        else:
            return 0.

    def __setitem__(self, key, value):
        self.data[key] = value

    def __len__(self):
        return self.data.size

    @property
    def seconds(self):
        return len(self) / self.samplerate

    def copy(self):
        return Sound(data = self.data, samplerate = self.samplerate)

    def sub_sound(self, start_index = 0, end_index = None):
        # if end_index > len(self), end_index = len(self) because of how slicing works
        if end_index is None:
            end_index = len(self)
        if start_index > end_index:
            raise ValueError(f"start_index {start_index} is greater than end_index {end_index}")
        return Sound(data = self.data[start_index : end_index], samplerate = self.samplerate)

    def read(self, filename):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.samplerate, self.data = wavfile.read(filename + (".wav" if not filename.endswith(".wav") else ""))
        self.data = self.data.astype(np.float32)
        if self.data.ndim != 1:
            self.data = np.mean(self.data, 1)

    def save(self, filename, clip = False):
        if clip:
            self.distort(1)
        if not filename.endswith(".wav"):
            filename += ".wav"
        wavfile.write(filename, self.samplerate, self.data.astype(np.float32))

    def play(self, sync = True):
        if os.name == "nt":
            self.save("__temp__.wav")
            utils.play_file("__temp__.wav", sync)
            if sync:
                os.remove("__temp__.wav")
        else:
            print("Warning: Sound.play only works on Windows")

    def show(self):
        if os.name != "nt":
            print("Warning: Sound.show only works on Windows")
            return
        plt.title("Sound")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.ylim(-1.1, 1.1)
        time = len(self) / self.samplerate
        time_axis = np.linspace(0, time, len(self))
        plt.plot(time_axis, self.data, linewidth = 0.75)
        plt.plot([0., time], [0., 0.], "k--", linewidth = 0.5)
        plt.plot([0., time], [1, 1], "r--", linewidth = 0.5)
        plt.plot([0., time], [-1, -1], "r--", linewidth = 0.5)
        plt.show()

    def fft(self):
        return rfft(self.data)

    def ifft(self, transform):
        self.data = irfft(transform)

    def norm_fft(self):
        # returns simple frequency spectrum
        # x[0] = DC, x[-1] = Nyquist frequency
        # output values = amplitude of frequency
        transform = np.abs(self.fft()) / len(self) # normalize fft
        transform[1:] *= 2 # scale except DC component
        return transform

    @property
    def fundamental(self):
        # maybe use cepstrum?
        # currently, finds first spectrum peak greater than 1/4 of highest peak
        transform = np.abs(self.fft())
        threshold = max(transform) / 4
        for i in range(1, len(transform)): # exclude DC component
            if (transform[i] > transform[i - 1]) and (transform[i] > transform[i + 1]):
                if transform[i] > threshold:
                    return i / transform.size * self.samplerate / 2
        raise ZeroDivisionError # because idk what went wrong

    def show_fft(self):
        transform = self.norm_fft()
        time_axis = np.linspace(0, self.samplerate / 2, transform.size)
        plt.xlabel("Frequency")
        plt.ylabel("Amplitude")
        plt.plot(time_axis, transform)
        plt.show()

    @property
    def amplitude(self):
        return np.maximum(self.data.max(), -self.data.min())

    @property
    def rms(self):
        return np.sqrt(np.mean(np.square(self.data)))

    def sine(self, frequency, amplitude = 1):
        self.data += np.sin(
            np.linspace(0, frequency * 2 * np.pi * len(self) / self.samplerate, len(self))) * amplitude

    def square(self, frequency, amplitude = 1):
        self.data += sig.square(
            np.arange(len(self)) * 2 * np.pi * frequency / self.samplerate) * amplitude

    def sawtooth(self, frequency, amplitude = 1):
        self.data += sig.sawtooth(
            np.arange(len(self)) * 2 * np.pi * frequency / self.samplerate + np.pi) * amplitude

    def triangle(self, frequency, amplitude = 1):
        self.data += sig.sawtooth(
            np.arange(len(self)) * 2 * np.pi * frequency / self.samplerate + np.pi / 2, 0.5) * amplitude

    def chirp(self, freq_start, freq_end, exponent = 1, amplitude = 1):
        # y = sin(tau * (c * x^z + f0*x))
        #     where c = (f1-f0) / ((z+1) * t^z))
        # frequency(x) = (f1-f0) * (x/t)^z + f0
        c = (freq_end - freq_start) / ((exponent + 1) * self.seconds ** exponent)
        x = np.linspace(0, self.seconds, len(self))
        t = c * x ** (exponent + 1) + freq_start * x
        self.data += np.sin(tau * t) * amplitude

    def noise(self, amplitude = 1):
        self.data += np.random.default_rng().uniform(-1, 1, len(self)) * amplitude

    def polygon(self, frequency, amplitude, num_sides, phase_shift = 0):
        # https://www.desmos.com/calculator/cgvpvyoqgl
        angle = 2 * np.pi / num_sides
        p = phase_shift * angle
        t = 2 * np.pi * frequency * np.linspace(0, len(self) / self.samplerate, len(self))
        u = 1 / np.tan(np.pi / num_sides)
        v = (t + p) % (angle)
        r = u / (np.sin(v) + u * np.cos(v))
        s = r * np.sin(t)
        self.data = s * amplitude

    def resize(self, newsize):
        if newsize > len(self):
            self.data = np.pad(self.data, (0, newsize - len(self)))
        else:
            self.data = self.data[:newsize]

    def append(self, sound2):
        self.data = np.append(self.data, sound2.data)

    def repeat(self, amount = 1):
        self.data = np.tile(self.data, amount)

    def mute(self):
        self.data[:] = 0.

    def trim_silence(self, threshold = 0.01, start = True, end = True):
        start_index = 0
        if start:
            for i in range(1, len(self)):
                if np.abs(self.data[i]) > threshold:
                    start_index = i - 1
                    break
        end_index = len(self)
        if end:
            for i in range(len(self) - 2, 0, -1):
                if np.abs(self.data[i] > threshold):
                    end_index = i + 1
                    break
        self.data = self.data[start_index:end_index]

    def set_at(self, sound2, offset = 0, multiplier = 1):
        limit = np.minimum(len(sound2), len(self) - offset)
        self.data[offset:offset + limit] = sound2.data[:limit] * multiplier

    def __add__(self, sound2):
        out = self.copy()
        out.add(sound2)
        return out

    def __mul__(self, factor):
        if factor == 1:
            return self.copy()
        out = Sound(data = self.data * factor, samplerate = self.samplerate)
        return out

    def add(self, sound2, offset = 0, multiplier = 1):
        limit = np.minimum(len(sound2), len(self) - offset)
        try:
            self.data[offset:offset + limit] += sound2.data[:limit] * multiplier
        except:
            print("Debug", len(self), offset, limit)
            raise ZeroDivisionError

    def invert(self):
        self.data = -self.data

    def reverse(self):
        self.data = np.flip(self.data)

    def amplify(self, factor):
        self.data *= factor

    def normalize(self, peak = 1):
        amplitude = self.amplitude
        if amplitude != 0:
            self.data *= peak / amplitude

    def normalize_rms(self, peak = 1):
        self.data *= peak / self.rms

    def distort(self, threshold = 1):
        self.data = np.clip(self.data, -threshold, threshold)

    def soft_clip(self, threshold = 1):
        # y = -t, x <= -t
        # y = (3x - x^3/t^2) / 2, -t < x < t
        # y = t, x >= t
        self.data = np.piecewise(self.data,
            [self.data <= -threshold, np.logical_and(self.data > -threshold, self.data < threshold), self.data >= threshold],
            [-threshold, lambda x: (3 * x - x ** 3 / threshold ** 2) / 2, threshold])

    def bit_crush(self, bits = 8):
        factor = np.float32(1 << (bits - 1))
        self.data = np.round(self.data * factor) / factor

    def power(self, exponent = 1):
        self.data = np.sign(self.data) * np.power(np.abs(self.data), exponent)

    def fade(self, start_index = 0, end_index = None, start_amp = 1, end_amp = 0., exponent = 1):
        if start_index < 0:
            start_index = 0
        end_index = end_index or len(self)
        if end_index > len(self): end_index = len(self)
        if start_index > end_index:
            raise ValueError(f"start_index {start_index} is greater than end_index {end_index}")
        numsamples = end_index - start_index
        self.data[start_index:end_index] *= np.linspace(start_amp, end_amp, numsamples) ** exponent

    def stretch(self, factor = 1, in_place = True):
        if not in_place:
            if factor == 1:
                return self
            return Sound(data = sig.resample(self.data, int(len(self) / factor)), samplerate = self.samplerate)
        self.data = sig.resample(self.data, int(len(self) / factor))

    def moving_average(self, amount = 2):
        # problem with padding? extra length added
        self.data = sig.convolve(self.data, np.repeat(1 / amount, amount))

    def convolve(self, kernel):
        if isinstance(kernel, Sound):
            kernel = kernel.data
        self.data = sig.convolve(self.data, kernel)

    def filter(self, type_, cutoff, order = 2):
        if type_ not in ["lp", "hp", "bp", "bs"]:
            raise ValueError(f"Invalid filter type: \"{type_}\"")
        sos = sig.butter(order, cutoff, type_, output = "sos", fs = self.samplerate)
        self.data = sig.sosfilt(sos, self.data)

    def filter_curve(self, response):
        # not perfect...
        if isinstance(response, Sound):
            kernel = response.ifft()
        else:
            kernel = Sound()
            kernel.ifft(np.array(response))
        self.convolve(kernel)

    def fft_filter(self, pass_low = None, pass_high = None):
        # brickwall bandpass filter using fft
        if pass_low is None:
            pass_low = 0
        if pass_high is None:
            pass_high = self.samplerate / 2
        bin_low = round(pass_low / self.samplerate * len(self))
        bin_high = round(pass_high / self.samplerate * len(self))
        ft = self.fft()
        ft[:bin_low] = 0
        ft[bin_high:] = 0
        self.ifft(ft)

    def reverb(self, delay_time = 0.1, decay = 0.7, mix = 1.):
        # Adapted from https://github.com/Rishikeshdaoo/Reverberator

        delay = int(delay_time * self.samplerate)

        c1 = utils.comb(self, decay, delay)
        c2 = utils.comb(self, decay - 0.1337, delay - 401)
        c3 = utils.comb(self, decay - 0.2718, delay + 801)
        c4 = utils.comb(self, decay - 0.3141, delay + 252)
        c = (c1 + c2 + c3 + c4) * 0.25

        a0 = utils.allpass(c, 0.7, 1051)
        a1 = utils.allpass(a0, 0.7, 337)
        a2 = utils.allpass(a1, 0.7, 113)

        self.amplify(1 - mix)
        self.add(a2 * mix)

    def conv_reverb(self, delay_time = 0.1, decay = 0.7, mix = 1.):
        # May be faster than reverb() for longer sounds
        # Create an impulse, apply reverb(), convolve with sound

        impulse = Sound(self.samplerate * 1.5) # 1.5 seconds is long enough
        for i in range(25):
            impulse[i] = 1 / (i + 1)

        impulse.reverb(delay_time, decay, 1.)
        impulse.normalize()

        wet = self.copy()
        wet.convolve(impulse)
        wet = wet * 0.4
        self.amplify(1 - mix)
        self.add(wet * mix)

    def autotune(self, window_size = 7000):
        out = Sound(len(self))
        sliced = np.array_split(self.data, np.ceil(len(self) / window_size))
        for i, part in enumerate(sliced):
            s = Sound(data = part)
            freq = s.fundamental
            newfreq = utils.nearest_note_frequency(freq)
            factor = newfreq / freq
            s.stretch(factor)
            s.fade(end_index = 50, start_amp = 0, end_amp = 1)
            s.fade(start_index = s.length - 50)
            out.set_at(s, window_size * i)
        return out

    def vocode(self, window_size = 7500, harmonics = 3, fade = 50):
        out = Sound(len(self))
        sliced = np.array_split(self.data, np.ceil(len(self) / window_size))
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

from . import utils
