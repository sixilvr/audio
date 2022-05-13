import numpy as np

from audio import Sound
from audio import Pattern
from audio import utils

class Arrangement(Pattern):
    def __init__(self, bpm, num_measures, beats_per_measure = 4, samplerate = 44100):
        super().__init__(bpm, num_measures * beats_per_measure, samplerate)
        self.beats_per_measure = beats_per_measure

    def __repr__(self):
        return f"audio.Arrangement(bpm = {self.bpm}, num_measures = {self.num_measures}, beats_per_measure = {self.beats_per_measure}, samplerate = {self.samplerate})"

    @property
    def measures(self):
        return utils.samples_to_beats(self.bpm, self.length, self.samplerate) / self.beats_per_measure

    def place_pattern(self, pattern, measure_location):
        self.place(pattern, (measure_location - 1) * self.beats_per_measure + 1)

    def repeat_pattern(self, pattern, start_measure, end_measure):
        repetitions = int((end_measure - start_measure + 1) / pattern.beats * self.beats_per_measure)
        self.roll(pattern, (start_measure - 1) * self.beats_per_measure + 1, repetitions, pattern.beats)
