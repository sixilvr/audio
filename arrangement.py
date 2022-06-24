import numpy as np

from . import Sound
from . import Pattern
from . import utils

class Arrangement(Pattern):
    def __init__(self, bpm, num_measures, beats_per_measure = 4):
        super().__init__(bpm, num_measures * beats_per_measure)
        self.beats_per_measure = beats_per_measure

    def __repr__(self):
        return f"audio.Arrangement(bpm = {self.bpm}, num_measures = {self.num_measures}, beats_per_measure = {self.beats_per_measure}, samplerate = {self.samplerate})"

    @property
    def measures(self):
        return utils.samples_to_beats(self.bpm, len(self), self.samplerate) / self.beats_per_measure

    def place_pattern(self, pattern, measure_location = 1, num_beats = None):
        if num_beats is None:
            self.place(pattern, (measure_location - 1) * self.beats_per_measure + 1)
        else:
            self.place(pattern.sub_pattern(1, num_beats), (measure_location - 1) * self.beats_per_measure + 1)

    def repeat_pattern(self, pattern, start_measure = 1, end_measure = -1, multiplier = 1):
        if end_measure == -1:
            end_measure = self.measures
        repetitions = int((end_measure - start_measure + 1) / pattern.beats * self.beats_per_measure)
        self.roll(pattern, (start_measure - 1) * self.beats_per_measure + 1, repetitions, pattern.beats, multiplier)
