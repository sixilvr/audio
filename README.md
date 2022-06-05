# audio

A Python module for interacting with audio data and creating music

Start by importing the audio module
```python
import audio as a
```

A Sound object can be made from a .wav file, existing sample data, or from scratch
```python
#creating from .wav file
x = a.Sound(file = "mysound.wav")

#creating from an existing array of samples
import numpy as np
sine_wave_data = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
x = a.Sound(data = sine_wave_data)

#creating a new Sound using the number of samples
x = a.Sound(44100) # sound is initialized to 44100 samples with a value of 0
```