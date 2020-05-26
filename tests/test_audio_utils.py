from ml_utilities import Audio
import numpy as np

def test_can_create_audio():
    audio = Audio()

    assert isinstance(audio, Audio)
    assert len(audio.data) == 0
    assert audio.samples is None

def test_can_load_audio():
    audio = Audio.load_audio('audio/20200428_091053.WAV')

    assert isinstance(audio,Audio)
    assert len(audio.data)  > 0
    assert audio.samples == len(audio.data)
    assert audio.sample_rate == 384000

def test_can_load_tone_sine():
    freq = 2 # Hz
    sample_rate = 50
    samples = sample_rate

    audio = Audio.load_tone(freq,samples,'sine',sample_rate)

    assert isinstance(audio, Audio)
    assert len(audio.data) == samples
    assert audio.data[0] == 0

def test_can_load_tone_square():
    freq = 2 # Hz
    sample_rate = 50
    samples = sample_rate

    audio = Audio.load_tone(freq,samples,'square',sample_rate)

    assert isinstance(audio, Audio)
    assert len(audio.data) == samples
    assert audio.data[1] == 1.0

def test_can_load_tone_sawtooth():
    freq = 2 # Hz
    sample_rate = 50
    samples = sample_rate

    audio = Audio.load_tone(freq,samples,'sawtooth',sample_rate)

    assert isinstance(audio, Audio)
    assert len(audio.data) == samples
    assert audio.data[0] == -1.0

def test_can_gain_mulitply():

    audio = Audio.load_tone(20,50,'square', 50)
    assert audio.data[0] == 1.0

    audio2 = audio.gain(0.5,'multiply')

    assert isinstance(audio, Audio)
    assert audio2.data[0] == 0.5

def test_can_gain_additive():
    audio = Audio.load_tone(20,50,'square', 50)
    assert audio.data[0] == 1.0

    audio2 = audio.gain(0.3,'additive')

    assert isinstance(audio, Audio)
    assert audio2.data[0] == 1.3

def test_can_gain_subtractive():
    audio = Audio.load_tone(20,50,'square', 50)
    assert audio.data[0] == 1.0

    audio2 = audio.gain(0.3,'subtractive')

    assert isinstance(audio, Audio)
    assert audio2.data[0] == 0.7

def test_spectrogram_1024():
    n_fft = 1024
    sample_rate = 250000
    samples = sample_rate * 1
    spectrogram = Audio.load_tone(20, samples,'sine', sample_rate).get_spectrogram_array_uint8(n_fft = n_fft)
    print(spectrogram.shape)

    assert spectrogram.shape[0] == (n_fft / 2) + 1, 'spectrogram height'
    assert spectrogram.shape[1] == int(samples * 2/ n_fft) + 1, 'spectrogram width'
    assert np.array([all(0 <= y <= 255 for y in x) for x in spectrogram]).all(), f'range check {spectrogram} failed. Expected all values 0-255'

def test_spectrogram_512():
    n_fft = 512
    sample_rate = 250000
    samples = sample_rate * 1
    spectrogram = Audio.load_tone(20, samples,'sine', sample_rate).get_spectrogram_array_uint8(n_fft = n_fft)
    print(spectrogram.shape)

    assert spectrogram.shape[0] == (n_fft / 2) + 1, 'spectrogram height'
    assert spectrogram.shape[1] == int(samples * 2 / n_fft)+ 1, 'spectrogram width'
    assert np.array([all(0 <= y <= 255 for y in x) for x in spectrogram]).all(), f'range check {spectrogram} failed. Expected all values 0-255'

def test_spectrogram_256():
    n_fft = 256
    sample_rate = 250000
    samples = sample_rate * 1
    spectrogram = Audio.load_tone(20, samples,'sine', sample_rate).get_spectrogram_array_uint8(n_fft = n_fft)

    assert spectrogram.shape[0] == (n_fft / 2) + 1, 'spectrogram height'
    assert spectrogram.shape[1] == int(samples * 2 / n_fft) + 1, 'spectrogram width'
    assert np.array([all(0 <= y <= 255 for y in x) for x in spectrogram]).all(), f'range check {spectrogram} failed. Expected all values 0-255'

