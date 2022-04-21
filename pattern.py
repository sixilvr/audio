import numpy as np

from audio.sound import Sound
from . import utils

class Pattern(Sound):
    def __init__(self, bpm = 120, beats = 8, samplerate = 44100):
        self.bpm = bpm
        self.samplerate = samplerate
        self.data = np.zeros(utils.tosamples(bpm, beats, samplerate), dtype = np.float32)

    @property
    def beats(self):
        return round(self.length * self.bpm / self.samplerate / 60., 3)

    def place(self, sound, beat = 1., multiplier = 1., stretch = 1., cut = False):
        place_func = self.set_at if cut else self.add
        sample_index = int(self.samplerate * 60 / self.bpm * (beat - 1.))
        sound2 = sound if stretch == 1. else sound.stretch(stretch, in_place = False)
        if cut:
            self.fade(start_index = sample_index - 200, end_index = sample_index)
        place_func(sound2, sample_index, multiplier)

    def roll(self, sound, beat, amount, interval, multiplier = 1., cut = False):
        for i in range(amount):
            self.place(sound, beat + i * interval, multiplier, cut = cut)

    def place_pattern(self, sound, pattern, beat_size = .5, cut = False, multiplier = 1., root_note = "C4", rest_char = 0):
        if len(pattern) * beat_size != self.beats:
            print("Invalid pattern length", self.beats, len(pattern) * beat_size, self.length, self.bpm)
            return
        for i in range(int(self.beats / beat_size)):
            if pattern[i] != rest_char:
                if pattern[i] == root_note:
                    self.place(sound, beat_size * i + 1, multiplier, cut = cut)
                else:
                    self.place(sound, beat_size * i + 1, multiplier, utils.transpose(root_note, pattern[i]), cut)

    def mute(self, startbeat = 1, endbeat = None):
        if endbeat is None:
            endbeat = self.beats
        self.data[utils.tosamples(self.bpm, startbeat - 1, self.samplerate):utils.tosamples(self.bpm, endbeat, self.samplerate)] = 0.

    def repeat(self, amount = 1):
        self.data = np.tile(self.data, amount)
