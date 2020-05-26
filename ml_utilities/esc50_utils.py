import os, sys, librosa, csv, parse
import numpy as np

audio_extensions = ('.wav')
meta_extensions = ('.csv')

by_name = lambda dir_entry : dir_entry.name
by_path = lambda dir_entry : dir_entry.path
file_with_extension = lambda dir_entry, ext : dir_entry.is_file() and dir_entry.name.endswith(ext)
filter_on_unique = lambda items, unique = True : np.unique(items) if unique else np.array(items)

class ESC50(object):
    def __init__(self, dataset_path):
        self.base_path = dataset_path
        self.audio_path = os.path.join(self.base_path, 'audio')
        self.meta_path = os.path.join(self.base_path, 'meta')
        self.meta, self.fieldnames = self.get_meta_file()
        self.audio_files = self.get_audio_files()
        self.validate_meta()

    def get_audio_files(self):
        ''' Returns a list of audio file dir_entry objects '''
        return sorted([x for x in os.scandir(self.audio_path) if file_with_extension(audio_extensions)], key = by_name)
    
    def get_meta_file(self):
        ''' Returns a dir_entry to the meta file '''
        meta_files = sorted([x for x in os.scandir(self.meta_path) if file_with_extension(meta_extensions)])
        return meta_files[0]

    def get_meta_data(self):
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
        return filter_on_unique([x['filename'] for x in self.meta], unique= unique)

    def get_folds(self, unique = False):
        return filter_on_unique([int(x['fold']) for x in self.meta], unique= unique)
    
    def get_targets(self, unique = False):
        return filter_on_unique([int(x['target']) for x in self.meta], unique= unique)

    def get_categories(self, unique = False):
        return filter_on_unique([x['category'] for x in self.meta], unique= unique)

    def get_esc10s(self, unique = False):
        return filter_on_unique([x['category'] for x in self.meta], unique= unique)
    
    def get_takes(self, unique = False):
        return filter_on_unique([x['category'] for x in self.meta], unique= unique)

    def validate_meta(self):
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

    def to_mnist(self, cache_path,  resolution_x, resolution_y):
        # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        # x_train = arrays of data in [0:255] range
        # y_train = label (class)
        return None

        
    
