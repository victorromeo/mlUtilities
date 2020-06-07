import os, math
import numpy as np
from scipy.signal import butter, lfilter, windows, sawtooth, square, sosfilt
from scipy.signal import spectrogram as linear_spectrogram
import matplotlib.pyplot as plt
import librosa, soundfile
from librosa.display import specshow
from IPython.display import Audio as AudioPlay
from ml_utilities.noise_utils import white, blue, pink, violet, brown

audio_extensions = ['.WAV','.wav']

# Operation labels which track what has happened to an audio sample
_trimOperation = lambda start, stop : f'trim [{str(start)}:{str(stop)}]'
_resampleOperation = lambda original, target : f'resample [{str(original)}:{target}]'
_butterOperation = lambda lowcut, highcut, order : f'butter [{str(lowcut)}:{highcut}:{order}]'
_mixOperation = lambda a,b,left,right,mode : f'mix [{a.source_path}*{left},{b.source_path}*{right},{mode}]'
_noiseOperation = lambda color, samples : f'noise [{color}, {samples}]'
_toneOperation = lambda f, samples, shape : f'tone [{f},{samples},{shape}]'
_gainOperation = lambda level, mode : f'gain[{level},{mode}]'
_insertOperation = lambda a, b, offset, match : f'insert[{a.source_path},{b.source_path},{offset}, {match}]'

# Plot scaling helper for consistent looking graphs
_plotScaling = lambda data, sample_rate, scaling_factor=6 : [int((len(data) / sample_rate) * scaling_factor * 0.4), int((len(data) / sample_rate) * scaling_factor * 0.1)]

_cmap = 'magma'

def remove_extension(filename):
    for x in audio_extensions:
        filename = filename.replace(x, '')
    return filename

# TODO Move slice as a formal replacement for Trim with operation tracking
def slice_audio(data, start, stop, sample_rate):
    s = data[start:stop]
    start = start if start is not None and start >=0 and start < len(s) -1 else 0
    stop = stop if stop is not None and stop > 0 and stop < len(s) else len(s) - 1
    t = np.linspace(start/sample_rate, stop/sample_rate, len(s))
    return s, t

class Audio(object):
    operations:[] = []
    data:[] = np.array([])
    samples:int = None
    sample_rate:int = None

    def __init__(self):
        self.sample_rate = None
        self.samples = None

    @classmethod
    def load_audio(cls, source_path, start = 0, stop = None):
        assert os.path.exists(source_path), 'Audio file doesn\'t exist'

        o = cls()

        data, sample_rate = soundfile.read(source_path, start = start, stop = stop)
        
        operations = [f'load {source_path}']

        if start != 0 or stop != None:
            operations += [_trimOperation(start, stop)]

        o.populate(data, sample_rate, source_path, operations)

        return o

    @classmethod
    def load_noise(cls, color:str = 'white', samples = 0, sample_rate = 0):
        ''' Loads a noise profile into an Audio object '''

        assert color is not None and color in ['white','pink', 'blue', 'brown', 'violet'], 'Invalid color noise'
        assert samples > 0, 'Samples required'
        assert sample_rate > 0, 'Sample rate required'

        noise_generators = {
            'white': white,
            'pink': pink,
            'blue': blue,
            'brown': brown,
            'violet': violet
        }
        
        return cls().populate(noise_generators[color](samples), sample_rate, 'generated', [_noiseOperation(color, samples)])

    @classmethod
    def load_tone(cls, freq, samples:int, shape:str = 'sine', sample_rate:int = 0):
        ''' Generates a waveform with peak values between -1.0 and 1.0 '''

        assert samples is not None and samples > 0, 'samples required'
        assert freq is not None and freq > 0, 'freq required'
        assert sample_rate > 0, 'Sample rate required'
        assert shape in ['sine', 'sawtooth', 'square']

        length = samples / sample_rate

        t = np.linspace(0, length, int(samples))

        tone_generator = {
            'sine': lambda fs,t : np.sin(2 * np.pi * fs * t),
            'cosine': lambda fs,t : np.cos(2 * np.pi * fs * t),
            'sawtooth' : lambda fs,t: sawtooth(2 * np.pi * fs * t),
            'square' : lambda fs,t: square(2* np.pi * fs * t)
        }

        return cls().populate(tone_generator[shape](freq, t), sample_rate, 'generated', [_toneOperation(freq, samples, shape)])

    def dump(self, destination_path):
        ''' Writes the current audio to a files at the specified destination_path '''

        assert self.data is not None, 'Not loaded'
        soundfile.write(destination_path, self.data, self.sample_rate)

        return self

    def trim(self, start = 0, stop = None):
        ''' Trims the audio, into the specified length '''

        assert self.samples is not None and self.samples > 0, 'Not loaded'
        o = Audio()
        return o.populate(self.data[start:stop], self.sample_rate, self.source_path, self.operations + [_trimOperation(start,stop)])

    def resample(self, target_sample_rate:int):
        ''' Resamples (up or down) to the specified target_sample_rate '''

        assert self.samples is not None and self.samples > 0, 'Not loaded'
        return Audio().populate(librosa.resample(self.data,self.sample_rate, target_sample_rate, res_type='fft'), target_sample_rate, self.source_path, self.operations + [_resampleOperation(self.sample_rate, target_sample_rate)])

    def pitch_shift(self, n_steps:float = None, bins_per_octave = 12, res_type='kaiser_best'):
        ''' Shift the pitch of the waveform by n_steps semitones '''

        assert n_steps is not None, 'n_steps is required to pitch shift'
        return Audio().populate(librosa.effects.pitch_shift(self.data, sr = self.sample_rate, n_steps = n_steps), self.sample_rate, self.source_path, self.operations + [f'[pitch_shift {n_steps}]'])

    def time_stretch(self, rate: float = None):
        ''' Time-stretch an audio series by a fixed rate. '''

        assert rate is not None, 'rate is required'
        assert rate != 0, 'rate must not be zero'

        return Audio().populate(librosa.effects.time_stretch(self.data, rate), self.sample_rate, self.source_path, self.operations + [f'[time_stretch {rate}]'])

    def populate(self, data, sample_rate, source_path, operations = []):
        ''' Common utility method to store the audio data and calculate base details '''

        assert self.samples is None, f'Already loaded {self.samples}'
        assert data is not None, 'data required'
        assert len(data) > 0, 'data must not be empty'
        assert sample_rate is not None and sample_rate > 0, 'sample_rate required'
        assert source_path is not None and len(source_path) > 0, 'source_path required'
        
        self.data = data
        self.sample_rate = sample_rate
        self.source_path = source_path
        self.samples = len(data)
        self.duration = self.samples * 1.0 / sample_rate
        self.operations = operations

        return self

    def butter_bandpass_filter(self, lowcut, highcut, order=5):
        ''' Performs a butter bandpass filter on the current audio, returning the filtered audio as a new Audio object '''
        # Create the butter curve
        nyq = 0.5 * self.sample_rate
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], btype='bandpass', output='sos')
        
        # Apply the filter
        # data = lfilter(b, a, self.data)
        data = sosfilt(sos, self.data)

        return Audio().populate(data, self.sample_rate, self.source_path,self.operations + [_butterOperation(lowcut, highcut, order)])

    def gain(self, level, mode: 'multiply'):
        ''' Increases the volume of the audio by a specified amount '''

        amplifier = {
            'multiply' : lambda d, l: d * l,
            'percent' : lambda d,l: d * l/ 100.0,
            'additive' : lambda d,l: d + l,
            'subtractive' : lambda d,l: d - l,
            'dB_additive' : lambda d,l: librosa.db_to_amplitude((librosa.amplitude_to_db(d) + l)),
            'dB_subtractive' : lambda d,l: librosa.db_to_amplitude((librosa.amplitude_to_db(d) - l)),
            'dB_multiply' : lambda d,l: (10 ** (l/20))  * d
        }

        data = amplifier[mode](self.data, level)
        return Audio().populate(data, self.sample_rate, self.source_path, self.operations + [_gainOperation(level,  mode)])

    def mix(self, audio, left = 1.0, right = 1.0, mode:str = 'additive'):
        ''' Combines two audio records into a single source, with an optional mix rate '''

        assert audio is not None, 'Need audio to mix'
        assert audio.samples == self.samples, 'Sample length of both must be the same'

        mixers = {
            'additive': lambda x,y:x+y,
            'multiply': lambda x,y:x*y,
            'subtractive': lambda x,y:x-y
        }

        data = mixers[mode](self.data * left, audio.data * right)
        return Audio().populate(data, self.sample_rate, self.source_path, self.operations + [_mixOperation(self, audio, left, right, mode)])

    def plot_waveform(self, start=0, stop=None, to_file = None):
        ''' Plots an audio waveform of the current audio '''

        assert self.samples is not None and self.samples > 0, 'Not loaded'

        d, t = slice_audio(self.data, start, stop, self.sample_rate)

        plt.subplot(111)

        plt.plot(t, d)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.show()

    def plot_spectrogram(self, start:int = 0, stop:int = None, y_axis:str= 'linear', n_fft:int = 1024, hop_length = None, window = 'hann', cmap = _cmap, to_file:str = None, raw = False):
        ''' Plots a spectrogram with a linear frequency y-axis '''
        assert self.samples is not None and self.samples > 0, 'Not loaded'
        assert y_axis in ['linear','fft','hz','log','mel','cqt_hz','cqt_note'], f'Invalid y_axis: {y_axis}'

        d, t = slice_audio(self.data, start, stop,  self.sample_rate)
        
        D = librosa.amplitude_to_db(np.abs(librosa.stft(d, n_fft=n_fft, hop_length=hop_length, window=window)), ref=np.max)
        
        plt.subplot(111)
        librosa.display.specshow(D, y_axis=y_axis, sr=self.sample_rate, cmap = cmap, x_axis='time')
        
        if raw:
            plt.axis('off')
        else:
            plt.colorbar(format='%+2.0f db')
            plt.title(f'{os.path.basename(self.source_path)}[{start}:{stop}]\n{y_axis} window {window}')
            plt.xlabel('Time [sec]')
            plt.ylabel('Frequency [Hz]')

        if to_file is not None:
            plt.savefig(to_file, bbox_inches='tight' if raw else None, pad_inches = 0 if raw else None)
            plt.close()
        else:
            plt.show()
    
    def plot_linear_spectrogram(self, start:int = 0, stop:int = None, n_fft:int = 1024, hop_length = None, window = 'hann', cmap = _cmap, to_file:str = None, raw = False):
        self.plot_spectrogram(start,stop,'linear',n_fft,hop_length,window,cmap,to_file,raw)
    
    def plot_mel_spectrogram(self, start:int = 0, stop:int = None, n_fft:int= 1024, n_mels:int = 128, hop_length:int = None, to_file: str = None, cmap = _cmap):
        ''' Plots a spectrogram using a non-linear mel scale for the y-axis and power density for the peak color detail '''

        assert self.samples is not None and self.samples > 0, 'Not loaded'

        if hop_length is None:
            hop_length = n_fft / 2

        d, t = slice_audio(self.data, start, stop,  self.sample_rate)

        mels = librosa.feature.melspectrogram(d, int(self.sample_rate), n_fft=int(n_fft), n_mels= int(n_mels), hop_length=int(hop_length))
        mels = np.log(mels + 1e-9) # add small number to avoid log(0)
        
        plt.subplot(111)
        specshow(mels, x_axis='time', y_axis='mel', sr=self.sample_rate,  fmax=self.sample_rate / 2, cmap=cmap) 
        plt.title('Mel-frequency spectrogram in power density') 

        if to_file is not None:
            plt.savefig(to_file)
        else:
            plt.show()
    
    def plot_mel_dB_spectrogram(self, start:int = 0, stop:int = None, n_fft:int= 1024, n_mels:int = 128, hop_length:int = None, to_file: str = None, cmap = _cmap):
        ''' Plots a spectrogram using a non-linear mel scale for the y-axis, and dB scale for the peak color detail '''

        assert self.samples is not None and self.samples > 0, 'Not loaded'

        if hop_length is None:
            hop_length = n_fft / 2

        d, t = slice_audio(self.data, start, stop,  self.sample_rate)

        mels = librosa.feature.melspectrogram(d, int(self.sample_rate), n_fft=int(n_fft), n_mels= int(n_mels), hop_length=int(hop_length))
        mels = np.log(mels + 1e-9) # add small number to avoid log(0)

        S_dB = librosa.power_to_db(mels, ref=np.max) 

        plt.subplot(111)
        specshow(S_dB, x_axis='time', y_axis='mel', sr=self.sample_rate,  fmax=self.sample_rate / 2, cmap = cmap) 
        plt.title('Mel-frequency spectrogram in dB')
        plt.colorbar(format='%+2.0f dB') 
        
        if to_file is not None:
            plt.savefig(to_file)
        else:
            plt.show()

    def get_spectrogram_array(self, start:int = 0, stop:int = None, n_fft:int=1024, hop_length:int = None, mode: str = 'amplitude'):
        ''' Retrieve a Numpy array containing the STFT of the audio signal '''

        assert mode in ['amplitude', 'power', 'dB'], 'Expected amplitude, power or dB'

        y, t = slice_audio(self.data, start, stop,  self.sample_rate)

        if hop_length is None:
            hop_length = int(n_fft / 2)

        S = librosa.core.stft(y, n_fft = n_fft, hop_length = hop_length)
        
        if mode == 'amplitude':
            return S
        if mode == 'power':
            return S ** 2
        if mode == 'dB':
            return librosa.amplitude_to_db(S, ref=np.max)

    def get_mel_spectrogram_array(self, start:int = 0, stop:int = None, n_fft:int=1024, hop_length:int = None, n_mels:int = 60, mode:str = 'power'):

        ''' Retrieve a Numpy array containing the STFT of the audio signal in the mel scale '''
        assert mode in ['power', 'dB'], 'Expecting mode = \'power\' or \'dB\''

        d, t = slice_audio(self.data, start, stop, self.sample_rate)

        if hop_length is None:
            hop_length = int(n_fft / 2)
        
        S = librosa.feature.melspectrogram(d, sr= self.sample_rate, n_fft = n_fft, hop_length = hop_length, n_mels = n_mels)
        if mode == 'power':
            return S
        if mode == 'dB':
            s_DB = librosa.power_to_db(S, ref=np.max)
            return s_DB
        
        return None

    def get_sample_box(self, start_time:float, min_freq: float, width_time: float, height_freq: float):
        ''' Extracts a sample of an audio file, in a rectangular shape from the frequency-time domains '''

        start = int(start_time * self.sample_rate)
        stop = int((start_time + width_time) * self.sample_rate)

        audio_sample = self.trim(start, stop).butter_bandpass_filter(min_freq, min_freq + height_freq)
        return audio_sample

    def get_statistics(self):

        return {
            'Mean amplitude': self.data.mean(),
            'Max amplitude' : self.data.max(),
            'Min amplitude' : self.data.min(),
            'Samples': self.samples,
            'Sample rate': self.sample_rate,
            'Duration' : self.samples / self.sample_rate,
            'Mean norm': abs(self.data).mean(), # also known as DC offset
            'RMS amplitude': math.sqrt(np.sum(self.data ** 2) / self.samples)
        } 

    def insert(self, audio, start:int = 0, match_rms_amplitude = False):
        ''' Combines two audio sources, such that they are effectively added together in amplitude '''
        ''' Optional: match_rms_amplitude when True modifies the incoming audio amplitude to match the existing audio amplitude '''
        
        a = self.data
        if match_rms_amplitude:
            gain_factor = self.get_statistics()['RMS amplitude'] / audio.get_statistics()['RMS amplitude']
            b = audio.gain(gain_factor, 'multiply').data
        else:
            b = audio.data

        if len(a) + start < len(b):
            c = b.copy()
            c[start:len(a) + start] += a
        else:
            c = a.copy()
            c[start:len(b) + start] += b

        return Audio().populate(c, self.sample_rate, self.source_path, self.operations + [_insertOperation(self, audio, start, match_rms_amplitude)])
