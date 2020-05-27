from ml_utilities import Audio
import os, sys, librosa, csv, parse, pickle
import numpy as np
from tqdm.auto import tqdm

audio_extensions = ('.wav')
meta_extensions = ('.csv')

by_name = lambda dir_entry : dir_entry.name
by_path = lambda dir_entry : dir_entry.path
file_with_extension = lambda dir_entry, ext : dir_entry.is_file() and dir_entry.name.endswith(ext)
filter_on_unique = lambda items, unique = True : np.unique(items) if unique else np.array(items)

class ESC50(object):
    def __init__(self, dataset_path):
        ''' Initialises this representation of the ESC50 like repository '''
        self.base_path = dataset_path
        self.audio_path = os.path.join(self.base_path, 'audio')
        self.meta_path = os.path.join(self.base_path, 'meta')
        self.meta, self.meta_fieldnames = self.get_meta_data()
        self.audio_files = self.get_audio_files()
        self.validate_meta()

    def get_audio_files(self):
        ''' Returns a list of audio file dir_entry objects '''

        return sorted([x for x in os.scandir(self.audio_path) if file_with_extension(x, audio_extensions)], key = by_name)
    
    def get_audio(self, filename):
        filepath = os.path.join(self.audio_path, filename)
        if os.path.exists(filepath):
            return Audio.load_audio(filepath)

        return None

    def get_meta_file(self):
        ''' Returns a dir_entry to the meta file '''

        meta_files = sorted([x for x in os.scandir(self.meta_path) if file_with_extension(x, meta_extensions)])
        return meta_files[0]

    def get_meta_data(self):
        ''' Read the csv meta file '''
        meta_file = self.get_meta_file()

        meta = []
        with open(meta_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                meta.append(row)

            fieldnames = reader.fieldnames
        # filename,fold,target,category,esc10,src_file,take
        return meta, fieldnames

    def get_filenames(self, unique = False):
        ''' Return the 'filename' column only, from the csv meta file '''
        return filter_on_unique([x['filename'] for x in self.meta], unique= unique)

    def get_folds(self, unique = False):
        ''' Return the 'fold' column only, from the csv meta file '''
        return filter_on_unique([int(x['fold']) for x in self.meta], unique= unique)
    
    def get_targets(self, unique = False):
        ''' Return the 'target' (class) column only, from the csv meta file '''
        return filter_on_unique([int(x['target']) for x in self.meta], unique= unique)

    def get_categories(self, unique = False):
        ''' Return the 'category' column only, from the csv meta file '''
        return filter_on_unique([x['category'] for x in self.meta], unique= unique)

    def get_esc10s(self, unique = False):
        ''' Return the ESC10 column only, from the csv meta file '''
        return filter_on_unique([x['category'] for x in self.meta], unique= unique)
    
    def get_takes(self, unique = False):
        ''' Return the 'take' column only, from the csv meta file '''
        return filter_on_unique([x['category'] for x in self.meta], unique= unique)

    def validate_meta(self):
        ''' Iteratively validate the meta csv file to ensure the ESC50 records are accurate '''

        assert len(self.meta) > 0, 'No meta files were found'
        assert len(self.audio_files) <= len(self.meta), 'More audio than meta records were found'
        assert len(self.meta) <= len(self.audio_files), 'More meta records than audio were found' 
        
        file_pattern = parse.compile('{fold}-{source}-{take}-{target}.wav')
        for row in self.meta:
            attributes = file_pattern.parse(row["filename"])

            if attributes is None:
                print(f'{row["filename"]} not matching pattern <fold>-<source>-<take>-<target>.wav')
            else:
                attribute_keys = ['fold', 'target' , 'take']
                
                errors = []
                for key in attribute_keys:
                    if attributes.named[key] != row[key]:
                        errors.append(f'{key} expected {row[key]} got {attribute.named[key]}')

                if len(errors) > 0:
                    print(f'{row["filename"]} incorrect. {", ".join(errors)}')

    def to_mnist(self, train_folds=[], test_folds=[], cache_path:str = None, n_fft:int = 1024, hop_length:int = None, flatten = True):
        
        # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        # x_train = arrays of data in [0:255] range
        # y_train = label (class)
        
        #Setup Caching
        cache_training_fold: str = "".join([str(x) for x in train_folds])
        cache_testing_fold: str = "".join([str(x) for x in test_folds])

        cache_file = f'mnist_{str(n_fft)}_{str(hop_length)}_{str(int(flatten))}_{cache_training_fold}_{cache_testing_fold}.pkl'
        cache_filename = os.path.join(cache_path, cache_file)

        if cache_path is not None:
            if os.path.exists(cache_filename):
                with open(cache_filename, 'rb') as f:
                    return pickle.load(f) 

        if hop_length is None:
            hop_length = int(n_fft / 4)

        all_records = zip(self.get_folds(), self.get_filenames(), self.get_targets())
        
        x_train = []
        y_train = []
        
        x_test = []
        y_test = []

        shape = None
    
        for folds, x, y in tqdm([(train_folds, x_train, y_train), (test_folds, x_test, y_test)], desc='Fold'):
            for fold, filename, target in tqdm([r for r in all_records if int(r[0]) in folds], desc='File'):
                audio = self.get_audio(filename)

                if audio is None:
                    continue
                
                spectrogram = audio.get_spectrogram_array_uint8(n_fft = n_fft, hop_length = hop_length)
                shape = spectrogram.shape
                x.append(spectrogram.flatten() if flatten else spectrogram)
                y.append(target)

        if cache_path is not None:
            with open(cache_filename, 'wb') as f:
                pickle.dump((x_train, y_train, x_test,y_test, shape), f)

        return x_train, y_train, x_test, y_test, shape

    def generate_spectrograms(self, dest_path:str = None, n_fft = 1024, hop_length = None, cmap:str = 'gray_r', raw:bool=True, y_axis = 'linear', dest_exists_ok = False,  image_exists_mode = 'replace'):
        if dest_path is None:
            dest_path = os.path.join(self.audio_path, 'spectrograms')
        
        try:
            os.makedirs(dest_path, exist_ok=dest_exists_ok)
        except:
            raise Exception(f'Destination path already exists: {dest_path}')

        for audio_file in tqdm(self.audio_files):
            # Calculate the destination filename
            basename = audio_file.name.replace('.WAV','').replace('.wav','')
            dest_file = os.path.join(dest_path, f'{basename}.png')

            # Determine whether to proceed
            if os.path.exists(dest_file):
                if image_exists_mode == 'skip':
                    continue
                elif image_exists_mode == 'error':
                    raise Exception(f'Spectrogram already exists: {dest_file}')
            
            # Load Audio
            audio = self.get_audio(audio_file)
            if audio is None:
                continue
            
            # Generate Spectrogram
            audio.plot_spectrogram(to_file=dest_file, n_fft=n_fft, hop_length=hop_length, y_axis=y_axis, raw=True, cmap=cmap)
    
    def generate_jonnor_mel_spectrograms(self, cache_path:str = None, train_folds = [], test_folds = []):
        # https://github.com/jonnor/ESC-CNN-microcontroller/releases/download/print1/report-print1.pdf
        target_sample_rate = 22050
        melfilter_bands = 60
        fft_length = 1024
        fft_hop = 512
        flatten = True
    
        # Jonnor paper suggests using 12 variations of Data Augmentation
        # During preprocessing, Data Augmentation is also performed. Time-stretching and Pitchshifting was done following [74], for a total of 12 variations per sample. The preprocessed
        # mel-spectrograms are stored on disk as Numpy arrays for use during training

        # [74] https://arxiv.org/pdf/1608.04363.pdf
        # • Time Stretching (TS): slow down or speed up the audio
        # sample (while keeping the pitch unchanged). Each sample
        # was time stretched by 4 factors: {0.81, 0.93, 1.07, 1.23}.
        time_stretching = [0.81, 0.93, 1.07, 1.23]
 
        # • Pitch Shifting (PS1): raise or lower the pitch of the
        # audio sample (while keeping the duration unchanged).
        # Each sample was pitch shifted by 4 values (in semitones):
        # {−2, −1, 1, 2}.
        pitch_shifting_1 = [-2, -1, 1, 2]
 
        # • Pitch Shifting (PS2): since our initial experiments indicated that pitch shifting was a particularly beneficial
        # augmentation, we decided to create a second augmentation set. This time each sample was pitch shifted by 4
        # larger values (in semitones): {−3.5, −2.5, 2.5, 3.5}.
        pitch_shifting_2 = [-3.5, -2.5, 2.5, 3.5]
 
        # • Dynamic Range Compression (DRC): compress the
        # dynamic range of the sample using 4 parameterizations, 3
        # taken from the Dolby E standard [29] and 1 (radio) from
        # the icecast online radio streaming server [30]: {music
        # standard, film standard, speech, radio}.
        dynamic_range_compression = ['music standard', 'film standard', 'speech', 'radio']
 
        # • Background Noise (BG): mix the sample with another
        # recording containing background sounds from different
        # types of acoustic scenes. Each sample was mixed with
        # 4 acoustic scenes: {street-workers, street-traffic, streetpeople, park}
        # . Each mix z was generated using z =
        # (1−w)·x+w·y where x is the audio signal of the original
        # sample, y is the signal of the background scene, and w is
        # a weighting parameter that was chosen randomly for each
        # mix from a uniform distribution in the range [0.1, 0.5].
        background_noise = ['street-workers', 'street-traffic','streetpeople', 'park']

        # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        # x_train = arrays of data in [0:255] range
        # y_train = label (class)
        
        if fft_hop is None:
            fft_hop = int(n_fft / 4)

        #Setup Caching
        cache_training_fold: str = "".join([str(x) for x in train_folds])
        cache_testing_fold: str = "".join([str(x) for x in test_folds])

        cache_file = f'jonnor_{str(fft_length)}_{str(fft_hop)}_{str(int(flatten))}_{cache_training_fold}_{cache_testing_fold}.pkl'
        cache_filename = os.path.join(cache_path, cache_file)

        all_records = list(zip(self.get_folds(), self.get_filenames(), self.get_targets()))
        progress = []

        x_train = []
        y_train = []
        s_train = []

        x_test = []
        y_test = []
        s_test = []

        # Attempt cache load
        if cache_path is not None:
            os.makedirs(cache_path, exist_ok=True)
            if os.path.exists(cache_filename):
                with open(cache_filename, 'rb') as f:
                    x_train, y_train, s_train, x_test, y_test, s_test, progress = pickle.load(f) 
        
        def _generate_and_append(audio):
            spectrogram = audio.get_mel_spectrogram_array(n_fft = fft_length, hop_length = fft_hop, n_mels = melfilter_bands)
            
            x.append(spectrogram.flatten() if flatten else spectrogram)
            y.append(target)
            s.append(spectrogram.shape)

        def _save_pickle():
            if cache_path is not None:
                with open(cache_filename, 'wb') as f:
                    pickle.dump((x_train, y_train, s_train, x_test, y_test, s_test, progress), f)

        with tqdm(total = len(all_records), desc='File') as q:
            for folds, x, y, s in [(train_folds, x_train, y_train, s_train), (test_folds, x_test, y_test, s_test)]:
                for fold, filename, target in list([r for r in all_records if int(r[0]) in folds]):
                    if filename in progress:
                        q.update(1)
                        continue
                
                    audio = self.get_audio(filename)

                    if audio is None:
                        q.update(1)
                        continue
                    
                    _generate_and_append(audio)
                    for ts in time_stretching:
                        _generate_and_append(audio.time_stretch(rate = ts))
                    
                    for ps in pitch_shifting_1:
                        _generate_and_append(audio.pitch_shift(n_steps = ps))
                    
                    for ps in pitch_shifting_2:
                        _generate_and_append(audio.pitch_shift(n_steps = ps))

                    progress.append(filename)
                    _save_pickle()
                    q.update(1)

        return x_train, y_train, s_train, x_test, y_test, s_test
