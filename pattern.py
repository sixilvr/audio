import numpy as np

from audio import Sound
from audio import utils

class Pattern(Sound):
    def __init__(self, bpm = 120, num_beats = 8, samplerate = 44100):
        self.bpm = bpm
        self.samplerate = samplerate
        self.data = np.zeros(utils.beats_to_samples(bpm, num_beats, samplerate), dtype = np.float32)

    def __repr__(self):
        return f"audio.Pattern(bpm = {self.bpm}, num_beats = {self.num_beats}, samplerate = {self.samplerate})"

    @property
    def beats(self):
        return utils.samples_to_beats(self.bpm, self.length, self.samplerate)

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

    def place_pattern(self, sound, pattern, beat_size = 0.5, cut = False, multiplier = 1, root_note = "C4", rest_char = 0):
        if len(pattern) * beat_size != self.beats:
            raise ValueError(f"Invalid pattern length: expected {self.beats}, got {len(pattern) * beat_size} with beat size {beat_size}")
        for i in range(int(self.beats / beat_size)):
            if pattern[i] != rest_char:
                if pattern[i] == root_note:
                    self.place(sound, beat_size * i + 1, multiplier, cut = cut)
                else:
                    self.place(sound, beat_size * i + 1, multiplier, utils.transpose_factor(root_note, pattern[i]), cut)

    def place_midi(self, sound, pattern, beat_size = 0.5, cut = False, multiplier = 1, root_note = 60):
        self.place_pattern(sound, [utils.midi_to_note(i) if i != 0 else 0 for i in pattern], beat_size, cut, multiplier, utils.midi_to_note(root_note))

    def mute(self, startbeat = 1, endbeat = None):
        if endbeat is None:
            endbeat = self.beats
        self.data[utils.beats_to_samples(self.bpm, startbeat - 1, self.samplerate):utils.beats_to_samples(self.bpm, endbeat, self.samplerate)] = 0

    def repeat(self, amount = 1):
        self.data = np.tile(self.data, amount)
