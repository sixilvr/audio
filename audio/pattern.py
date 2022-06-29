import numpy as np

from . import Sound
from . import utils

class Pattern(Sound):
    def __init__(self, bpm = 120, num_beats = 8):
        self.bpm = bpm
        self.samplerate = 44100
        self.data = np.zeros(utils.beats_to_samples(bpm, num_beats, 44100), dtype = np.float32)

    def __repr__(self):
        return f"audio.Pattern(bpm = {self.bpm}, num_beats = {self.beats}, samplerate = {self.samplerate})"

    def copy(self):
        out = Pattern(self.bpm, self.beats)
        out.data = self.data
        return out

    @property
    def beats(self):
        return utils.samples_to_beats(self.bpm, len(self), self.samplerate)

    def sub_pattern(self, start = 1, end = None):
        if end is None:
            end = self.beats
        out = Pattern(self.bpm, end - start + 1, self.samplerate)
        out.data = self.data[utils.beats_to_samples(self.bpm, start - 1, self.samplerate):utils.beats_to_samples(self.bpm, end - 1, self.samplerate)]
        return out

    def place(self, sound, beat = 1, multiplier = 1, stretch = 1, cut = False):
        if beat < 1 or beat > self.beats + 1:
            return
        place_func = self.set_at if cut else self.add
        sample_index = utils.beats_to_samples(self.bpm, beat - 1, self.samplerate)
        sound2 = sound if stretch == 1. else sound.stretch(stretch, in_place = False)
        if cut and sample_index - 200 > 0:
            self.fade(start_index = sample_index - 200, end_index = sample_index)
        place_func(sound2, sample_index, multiplier)

    def roll(self, sound, beat, amount, interval, multiplier = 1, cut = False):
        for i in range(amount):
            self.place(sound, beat + i * interval, multiplier, cut = cut)

    def place_notes(self, sound, notes, beat_size = 0.5, cut = False, multiplier = 1, root_note = "C4", rest_char = 0):
        if len(notes) * beat_size != self.beats:
            raise ValueError(f"Invalid pattern length: expected {int(self.beats * beat_size)}, but got {int(len(notes) * beat_size)}")
        for i in range(int(self.beats / beat_size)):
            if notes[i] != rest_char:
                if notes[i] == root_note:
                    self.place(sound, beat_size * i + 1, multiplier, cut = cut)
                else:
                    self.place(sound, beat_size * i + 1, multiplier, utils.transpose_factor(root_note, notes[i]), cut)

    def place_midi(self, sound, pattern, beat_size = 0.5, cut = False, multiplier = 1, root_note = 60):
        self.place_notes(sound, [utils.midi_to_note(i) if i != 0 else 0 for i in pattern], beat_size, cut, multiplier, utils.midi_to_note(root_note))

    def place_hits(self, sound, pattern, beat_size = 0.5, cut = False, multiplier = 1, hit_char = "1"):
        if len(pattern) * beat_size != self.beats:
            raise ValueError(f"Invalid pattern length: this pattern is {self.beats}, but got {len(pattern) * beat_size} with beat size {beat_size}")
        if pattern == "0" * 16:
            return
        for i in range(int(self.beats / beat_size)):
            if pattern[i] == hit_char:
                self.place(sound, beat_size * i + 1, multiplier, cut = cut)

    def mute(self, startbeat = 1, endbeat = None):
        if endbeat is None:
            endbeat = self.beats
        self.data[utils.beats_to_samples(self.bpm, startbeat - 1, self.samplerate):utils.beats_to_samples(self.bpm, endbeat, self.samplerate)] = 0

    def repeat(self, amount = 1):
        self.data = np.tile(self.data, amount)
