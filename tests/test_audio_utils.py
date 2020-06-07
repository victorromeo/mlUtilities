from ml_utilities import Audio, to_uint8
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

def test_can_load_noise_white():
    sample_rate = 250000
    samples = sample_rate * 2
    audio = Audio.load_noise('white', samples, sample_rate)

    assert isinstance(audio,Audio)
    assert len(audio.data)  > 0
    assert audio.samples == len(audio.data) == samples
    assert audio.sample_rate == sample_rate

def test_can_load_noise_pink():
    sample_rate = 250000
    samples = sample_rate * 2
    audio = Audio.load_noise('pink', samples, sample_rate)

    assert isinstance(audio,Audio)
    assert len(audio.data)  > 0
    assert audio.samples == len(audio.data) == samples
    assert audio.sample_rate == sample_rate

def test_can_load_noise_blue():
    sample_rate = 250000
    samples = sample_rate * 2
    audio = Audio.load_noise('blue', samples, sample_rate)

    assert isinstance(audio,Audio)
    assert len(audio.data)  > 0
    assert audio.samples == len(audio.data) == samples
    assert audio.sample_rate == sample_rate

def test_can_load_noise_brown():
    sample_rate = 250000
    samples = sample_rate * 2
    audio = Audio.load_noise('brown', samples, sample_rate)

    assert isinstance(audio,Audio)
    assert len(audio.data)  > 0
    assert audio.samples == len(audio.data) == samples
    assert audio.sample_rate == sample_rate

def test_can_load_noise_violet():
    sample_rate = 250000
    samples = sample_rate * 2
    audio = Audio.load_noise('violet', samples, sample_rate)

    assert isinstance(audio,Audio)
    assert len(audio.data)  > 0
    assert audio.samples == len(audio.data) == samples
    assert audio.sample_rate == sample_rate


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

def test_can_gain_multiply():

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
    spectrogram = Audio.load_tone(20, samples,'sine', sample_rate).get_spectrogram_array(n_fft = n_fft)
    spectrogram = to_uint8(spectrogram)
    print(spectrogram.shape)

    assert spectrogram.shape[0] == (n_fft / 2) + 1, 'spectrogram height'
    assert spectrogram.shape[1] == int(samples * 2/ n_fft) + 1, 'spectrogram width'
    assert np.array([all(0 <= y <= 255 for y in x) for x in spectrogram]).all(), f'range check {spectrogram} failed. Expected all values 0-255'

def test_spectrogram_512():
    n_fft = 512
    sample_rate = 250000
    samples = sample_rate * 1
    spectrogram = Audio.load_tone(20, samples,'sine', sample_rate).get_spectrogram_array(n_fft = n_fft)
    spectrogram = to_uint8(spectrogram)

    print(spectrogram.shape)

    assert spectrogram.shape[0] == (n_fft / 2) + 1, 'spectrogram height'
    assert spectrogram.shape[1] == int(samples * 2 / n_fft)+ 1, 'spectrogram width'
    assert np.array([all(0 <= y <= 255 for y in x) for x in spectrogram]).all(), f'range check {spectrogram} failed. Expected all values 0-255'

def test_spectrogram_256():
    n_fft = 256
    sample_rate = 250000
    samples = sample_rate * 1
    spectrogram = Audio.load_tone(20, samples,'sine', sample_rate).get_spectrogram_array(n_fft = n_fft)
    spectrogram = to_uint8(spectrogram)

    assert spectrogram.shape[0] == (n_fft / 2) + 1, 'spectrogram height'
    assert spectrogram.shape[1] == int(samples * 2 / n_fft) + 1, 'spectrogram width'
    assert np.array([all(0 <= y <= 255 for y in x) for x in spectrogram]).all(), f'range check {spectrogram} failed. Expected all values 0-255'

def test_mel_spectrogram_1024():
    n_fft = 1024
    hop_length = 512
    n_mels = 60

    sample_rate = 384000
    samples = sample_rate * 1

    spectrogram = Audio.load_tone(20, samples,'sine', sample_rate).get_mel_spectrogram_array(n_fft = n_fft, n_mels=n_mels, hop_length=hop_length)
    spectrogram = to_uint8(spectrogram)
    print(spectrogram.shape)

    assert spectrogram.shape[0] == n_mels, 'spectrogram height'
    assert spectrogram.shape[1] == int(samples * 2/ n_fft) + 1, 'spectrogram width'
    assert np.array([all(0 <= y <= 255 for y in x) for x in spectrogram]).all(), f'range check {spectrogram} failed. Expected all values 0-255'


def test_mel_spectrogram_512():
    n_fft = 512
    hop_length = 256
    n_mels = 60

    sample_rate = 384000
    samples = sample_rate * 1

    spectrogram = Audio.load_tone(20, samples,'sine', sample_rate).get_mel_spectrogram_array(n_fft = n_fft, n_mels=n_mels, hop_length=hop_length)
    spectrogram = to_uint8(spectrogram)
    print(spectrogram.shape)

    assert spectrogram.shape[0] == n_mels, 'spectrogram height'
    assert spectrogram.shape[1] == int(samples * 2/ n_fft) + 1, 'spectrogram width'
    assert np.array([all(0 <= y <= 255 for y in x) for x in spectrogram]).all(), f'range check {spectrogram} failed. Expected all values 0-255'

def test_mel_spectrogram_256():
    n_fft = 256
    hop_length = 128
    n_mels = 60

    sample_rate = 384000
    samples = sample_rate * 1

    spectrogram = Audio.load_tone(20, samples,'sine', sample_rate).get_mel_spectrogram_array(n_fft = n_fft, n_mels=n_mels, hop_length=hop_length)
    spectrogram = to_uint8(spectrogram)
    print(spectrogram.shape)

    assert spectrogram.shape[0] == n_mels, 'spectrogram height'
    assert spectrogram.shape[1] == int(samples * 2/ n_fft) + 1, 'spectrogram width'
    assert np.array([all(0 <= y <= 255 for y in x) for x in spectrogram]).all(), f'range check {spectrogram} failed. Expected all values 0-255'

def test_pitch_shift_plus_one():
    ''' Raise the pitch of the whole audio file, one tone '''
    sample_rate = 250000
    samples = sample_rate * 1

    tone = Audio.load_tone(20, samples,'square', sample_rate=sample_rate).pitch_shift(1)

    assert tone is not None, 'Pitch shift failed'

def test_pitch_shift_minus_one():
    ''' Drop the pitch of the whole audio file, one tone '''
    sample_rate = 250000
    samples = sample_rate * 1

    tone = Audio.load_tone(20, samples,'square', sample_rate=sample_rate).pitch_shift(-1)

    assert tone is not None, 'Time stretch failed'
    assert tone.data.shape[0] == samples

def test_time_stretch_double_time():
    ''' Causes the audio to play back twice its current speed '''

    sample_rate = 250000
    samples = sample_rate * 1

    tone = Audio.load_tone(20, samples,'square', sample_rate=sample_rate).time_stretch(2.0)

    assert tone is not None, 'Time stretch failed'
    assert tone.data.shape[0] == samples / 2.0

def test_time_stretch_half_time():
    ''' Causes the audio to play back twice as slow as its current speed '''
    sample_rate = 250000
    samples = sample_rate * 1

    tone = Audio.load_tone(20, samples,'square', sample_rate=sample_rate).time_stretch(0.5)

    assert tone is not None, 'Time stretch failed'
    assert tone.data.shape[0] == samples / 0.5