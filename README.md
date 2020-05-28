# mlUtilities

A suite of common utilities to support machine learning data collation, inference and visualisation

## Background

To validate datasets, prepare new datasets and generate fake data, this collection of utilities assists Audio analysis. Examples to demonstrate use of functions listed below include:

## Examples

### Example of MNIST generation from ESC50

```python
import os
if not os.path.exists('ml_utilities')
    !git clone https://github.com/victorromeo/mlUtilities.git ml_utilities
if not os.path.exists('ml_utilities/sets/ESC50')
    !git clone https://github.com/karolpiczak/ESC-50.git ml_utilities/sets/ESC50

!pip install -q -r ml_utilities/requirements.txt
from ml_utilities.esc50_utils import ESC50

esc50 = ESC50('ml_utilities/sets/ESC50')
esc50.to_mnist()
```

### Example of ESC50 Jonnor preset MNIST conversion from ESC50

Uses preset of 22050 Hz sample rate, 60 melfilter banks, 1024 fft window length, 512 fft hop, 12 data augmentations including time_stretching and pitch shifting

```python
import os

if not os.path.exists('ml_utilities')
    !git clone https://github.com/victorromeo/mlUtilities.git ml_utilities
if not os.path.exists('ml_utilities/sets/ESC50')
    !git clone https://github.com/karolpiczak/ESC-50.git ml_utilities/sets/ESC50
!pip install -q -r ml_utilities/requirements.txt
from ml_utilities.esc50_utils import ESC50

preprocessed = esc50.generate_jonnor_mel_spectrograms(cache_path='/Volumes/Samsung_T5/tests/')
x_train, y_train, s_train, x_test, y_test, s_test = esc50.generate_jonnor_mnist(preprocessed, train_folds= [1,2,3,4], test_folds=[5])
```

### Notebooks for worked examples

- [audio_examples.ipynb](https://github.com/victorromeo/mlUtilities/blob/master/audio_examples.ipynb)
- [deepsqueak_examples.ipynb](https://github.com/victorromeo/mlUtilities/blob/master/deepsqueak_examples.ipynb)
- [esc50_examples.ipynb](https://github.com/victorromeo/mlUtilities/blob/master/esc50_examples.ipynb)
- [jonnor_examples.ipynb](https://github.com/victorromeo/mlUtilities/blob/master/jonnor_examples.ipynb)

## Contents

|Module|Object|Operation| Description |
|------|------|---------|-------------|
| ml_utilities.activation_utils | - | _see file_ | Various ML activation functions, for reference and experimentation |
| ml_utilities.audio_utils| Audio | load_audio | Load an audio file |
| ml_utilities.audio_utils| Audio | load_noise | Generate audio noise (white, pink, blue, brown, violet) |
| ml_utilities.audio_utils| Audio | load_tone  | Generate audio tones (sine, square, sawtooth) |
| ml_utilities.audio_utils| Audio | dump | Writes audio file in WAV format to disk |
| ml_utilities.audio_utils| Audio | trim | Truncates the audio signal |
| ml_utilities.audio_utils| Audio | gain | Modify signal amplitude (multiply, additive, subtractive, ...) |
| ml_utilities.audio_utils| Audio | resample | Resample the audio signal to a new sample rate |
| ml_utilities.audio_utils| Audio | time_stretch | Modify speed of audio, but not frequency |
| ml_utilities.audio_utils| Audio | pitch_shift | Modify frequency of audio, but not speed |
| ml_utilities.audio_utils| Audio | mix | Combine two equal length signals |
| ml_utilities.audio_utils| Audio | insert | Add a signal into another signal, matching target amplitude |
| ml_utilities.audio_utils| Audio | get_sample_box | Fetch a sample of audio in a time,frequency domain bounding box |
| ml_utilities.audio_utils| Audio | plot_waveform | Generate waveform, to display or disk |
| ml_utilities.audio_utils| Audio | plot_spectrogram | Generate spectrogram, to display or disk |
| ml_utilities.audio_utils| Audio | plot_linear_spectrogram | Generate linear y-axis spectrogram, to display or disk |
| ml_utilities.audio_utils| Audio | plot_mel_spectrogram | Generate mel scale y-axis spectrogram, to display or disk |
| ml_utilities.audio_utils| Audio | plot_mel_dB_spectrogram | Generate mel Db scale y-axis spectrogram, to display or disk |
| ml_utilities.audio_utils| Audio | get_spectrogram_array | Generate numpy spectrogram array in linear, or mel format |
| ml_utilities.audio_utils| Audio | get_spectrogram_array_uint8 | Generate numpy spectrogram array with amplitude 0-255 in linear, or mel format |
| ml_utilities.audio_utils| Audio | get_statistics | Gets the max,min,mean,samples, sample_rate,duration, mean_norm and RMS amplitude of signal |
| ml_utilities.deepsqueak_utils| - | read_calls  | Internal operation to parse a DeepSqueak Matlab mat detection file using h5py |
| ml_utilities.deepsqueak_utils| - | read_matlab  | Internal operation to read a Matlab mat file using h5py |
| ml_utilities.deepsqueak_utils| DeepSqueak | get_audio  | Fetches an Audio object for a specified DeepSqueak call |
| ml_utilities.deepsqueak_utils| DeepSqueak | get_call  | Fetches a single call record from a DeepSqueak detection mat file  |
| ml_utilities.deepsqueak_utils| DeepSqueak | get_calls  | Fetches a count of all calls from a DeepSqueak detection mat file |
| ml_utilities.moth_utils| - | datetime_to_datetime_string | Creates a datetime string which conforms to AudioMoth standards |
| ml_utilities.moth_utils| - | parse_filename_to_datetime_local | Creates an tz aware datetime from an AudioMoth filename (local TZ) |
| ml_utilities.moth_utils| - | parse_filename_to_datetime_utc | Creates an tz aware datetime from an AudioMoth filename (utc TZ) |
| ml_utilities.moth_utils| - | parse_filename_to_timestamp | Creates a timestamp from an AudioMoth filename |
| ml_utilities.moth_utils| - | rename_files_local_to_utc | Modifies the filename to change timezone of audio from local to utc  |
| ml_utilities.moth_utils| - | rename_files_utc_to_local | Modifies the filename to change timezone of audio from utc to local  |
| ml_utilities.esc50_utils| ESC50 | generate_jonnor_mel_spectrograms | Using a preset, creates a preprocessed data set containing mel spectrograms as numpy arrays from the ESC50 repository |
| ml_utilities.esc50_utils| ESC50 | generate_jonnor_mnist | Converts the output of the generate_jonnor_mel_spectrograms preprocessing step, to generate an MNIST dataset from ESC50 |
| ml_utilities.esc50_utils| ESC50 | generate_spectrograms | Generates linear or mel spectrograms for the ESC50 repository |
| ml_utilities.esc50_utils| ESC50 | get_audio | Fetches the Audio object for an ESC50 audio entry |
| ml_utilities.esc50_utils| ESC50 | get_audio_files | Retrieves a sorted list of DirEntry objects to audio files, for the ESC50 repository |
| ml_utilities.esc50_utils| ESC50 | get_categories | Retrieves all the categories from the meta CSV file, optional unique flag |
| ml_utilities.esc50_utils| ESC50 | get_esc10s | Retrieves all the esc10 reference from the meta CSV file, optional unique flag |
| ml_utilities.esc50_utils| ESC50 | get_filenames | Retrieves all the filename from the meta CSV file, optional unique flag |
| ml_utilities.esc50_utils| ESC50 | get_folds | Retrieves all the fold values from the meta CSV file, optional unique flag |
| ml_utilities.esc50_utils| ESC50 | get_meta_data | Internal operation to read the metadata from the ESC50 csv file |
| ml_utilities.esc50_utils| ESC50 | get_meta_file | Attempts to detect the meta CSV file for the ESC50 repository |
| ml_utilities.esc50_utils| ESC50 | get_takes | Retrieves all the take values from the meta CSV file, optional unique flag |
| ml_utilities.esc50_utils| ESC50 | get_targets | Retrieves all the target values from the meta CSV file, optional unique flag |
| ml_utilities.esc50_utils| ESC50 | to_mnist | Simplified one operation tranformation into MNIST format |
| ml_utilities.esc50_utils| ESC50 | validate_meta | Iteratively checks the metadata to ensure that audio and metadata match |
| ml_utilities.noise_utils | - | blue | Generates n samples of blue noise (+6dB power/octave : +3dB Power density/octave) |
| ml_utilities.noise_utils | - | brown | Generates n samples of brown noise (-3dB power/octave : -6dB Power density/octave) |
| ml_utilities.noise_utils | - | pink | Generates n samples of pink noise (+0dB power/octave : -3dB Power density/octave) |
| ml_utilities.noise_utils | - | violet | Generates n samples of violet noise (+9dB power/octave : +6dB Power density/octave) |
| ml_utilities.noise_utils | - | white | Generates n samples of white noise (+3dB power/octave : +0dB Power density/octave) |
| ml_utilities.noise_utils | - | mean_square | Calculates the mean square of a signal (mean of square of values) |
| ml_utilities.noise_utils | - | normalise | Normalised a signal to ensure all values are within a predefined range |

## Dependencies include

h5py==2.10.0
numpy==1.18.3
librosa==0.7.2
mat73==0.40
parse==1.15.0
pytest==5.4.2
scikit-learn==0.23.0
sox==1.3.7
tqdm==4.46.0
