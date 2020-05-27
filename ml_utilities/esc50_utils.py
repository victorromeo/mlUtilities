from ml_utilities import Audio
import os, sys, librosa, csv, parse, pickle
import numpy as np
import tqdm as tqdm

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

        all_records = list(zip(self.get_folds(), self.get_filenames(), self.get_targets()))
        
        x_train = []
        y_train = []
        
        x_test = []
        y_test = []
        
        shape = None
    
        for folds, x, y in [(train_folds, x_train, y_train), (test_folds, x_test, y_test)]:
            print(f'folds {folds} {x} {y}')
            for fold, filename, target in [r for r in all_records if r[0] in folds]:
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

        for audio_file in tqdm.tqdm(self.audio_files):
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
