import mat73, h5py
from ml_utilities import Audio

def read_matlab(mat_path):
    def conv(path=''):
        p = path or '/'
        paths[p] = ret = {}
        for k, v in f[p].items():
            if type(v).__name__ == 'Group':
                ret[k] = conv(f'{path}/{k}')  # Nested struct
                continue
            v = v[()]  # It's a Numpy array now
            if v.dtype == 'object':
                # HDF5ObjectReferences are converted into a list of actual pointers
                ret[k] = [r and paths.get(f[r].name, f[r].name) for r in v.flat]
            else:
                # Matrices and other numeric arrays
                ret[k] = v if v.ndim < 2 else v.swapaxes(-1, -2)
        return ret

    paths = {}
    with h5py.File(mat_path, 'r') as f:
        return conv()

def read_calls(mat_path):
    r = read_matlab(mat_path)

    def array_to_string(y):
        try:            
            return ''.join([chr(x) for x in y[0]])
        except Exception as e:
            print(f'error: {e}')
            return ''

    rate_ind, box_ind, box_rel_ind, score_ind, i4, i5, power_ind, accept_ind = [x.replace('/#refs#/','') for x in r['#refs#']['f']]

    Calls = {      
        'calls':r['#refs#'][rate_ind].shape[0],
        'boxes':r['#refs#'][box_ind],
        'boxes_rel':r['#refs#'][box_rel_ind],
        'power': r['#refs#'][power_ind][...,0],
        'scores': r['#refs#'][score_ind][...,0],
        'accepts':r['#refs#'][accept_ind][...,0],
        'audio_file':array_to_string(r['AudioFile']),
        'sample_rate':r['#refs#'][rate_ind][0,0],
        'detect_file':mat_path,
        'settings':{
            'setting0' : r['Settings'][0][0],
            'window_sec' : r['Settings'][1][0],
            'overlap_sec' : r['Settings'][2][0],
            'freq_max' : r['Settings'][3][0],
            'freq_min' : r['Settings'][4][0],
            'threshold' : r['Settings'][5][0],
            'setting6' : r['Settings'][6][0],
        },
        'networks':r['networkselections'],
        'detected':array_to_string(r['detectiontime'])
    }

    return Calls

class DeepSqueak():
    def __init__(self, mat_path, audio_path):
        self.mat_path = mat_path
        self.audio_path = audio_path
        self.Calls = read_calls(mat_path)
        self.audio = Audio.load_audio(audio_path)

    def get_call_count(self):
        return self.Calls['calls']

    def get_call(self, n:int):
        box = self.Calls['boxes'][n]
        power = self.Calls['power'][n]
        score = self.Calls['scores'][n]
        accept = self.Calls['accepts'][n]

        return box, power, score, accept

    def get_audio(self, n: int):
        box, power, score, accept = self.get_call(n)
        start_time, min_freq, width_time, height_freq = box

        return self.audio.get_sample_box(start_time, min_freq* 1000, width_time, height_freq * 1000)
    

