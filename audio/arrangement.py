from . import Pattern

class Arrangement(Pattern):
    def __init__(self, bpm, num_patterns, beats_per_pattern = 8):
        super().__init__(bpm, num_patterns * beats_per_pattern)
        self.beats_per_pattern = beats_per_pattern

    def __repr__(self):
        return f"audio.Arrangement(bpm = {self.bpm}, num_patterns = {self.num_patterns}, beats_per_pattern = {self.beats_per_pattern})"

    @property
    def num_patterns(self):
        return self.beats / self.beats_per_pattern

    def place_pattern(self, pattern, pattern_location = 1, num_beats = None):
        if num_beats is None:
            self.place(pattern, (pattern_location - 1) * self.beats_per_pattern + 1)
        else:
            self.place(pattern.sub_pattern(1, num_beats), (pattern_location - 1) * self.beats_per_pattern + 1)

    def repeat_pattern(self, pattern, start_location = 1, end_location = -1, multiplier = 1):
        if end_location == -1:
            end_location = self.num_patterns
        repetitions = int((end_location - start_location + 1) * self.beats_per_pattern / pattern.beats)
        self.roll(pattern, (start_location - 1) * self.beats_per_pattern + 1, repetitions, pattern.beats, multiplier)

    def arrange_pattern(self, pattern, arrangement_pattern, hit_char = "1"):
        if len(arrangement_pattern) != self.num_patterns:
            raise ValueError(f"Invalid arrangement pattern length: expected {self.num_patterns}, but got {len(arrangement_pattern)}")
        for i, char in enumerate(arrangement_pattern):
            if char == hit_char:
                self.place_pattern(pattern, i + 1)
