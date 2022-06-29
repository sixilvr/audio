# audio

## A Python module for interacting with audio data and creating music

### Installation
```bash
pip install git+https://github.com/sixilvr/audio.git
```

### Getting Started

Import the audio module
```python
import audio as a
```

This library includes three classes: Sound, Pattern, and Arrangement

### audio.Sound
A Sound object is used for audio data without any tempo/beat information.
It can be created from a .wav file, existing sample data, or be initialized to silence.
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

Sound, Pattern, and Arrangement can all be saved as a .wav file using the .save method
```python
import numpy as np
sine_wave_data = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
x = a.Sound(data = sine_wave_data)
x.save("filename.wav")
```
