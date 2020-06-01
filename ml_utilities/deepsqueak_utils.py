import mat73, h5py, os
from ml_utilities import Audio
import matplotlib.pyplot as plt
import tqdm.auto as tqdm

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
    audio:Audio = None

    def __init__(self, mat_path, audio_path):
        self.mat_path = mat_path
        self.audio_path = audio_path
        self.Calls = read_calls(mat_path)
        self.audio = None

    def get_call_count(self):
        return self.Calls['calls']

    def get_call(self, n:int):
        box = self.Calls['boxes'][n]
        power = self.Calls['power'][n]
        score = self.Calls['scores'][n]
        accept = self.Calls['accepts'][n]

        return box, power, score, accept

    def get_audio(self):
        if self.audio is None:
            self.audio = Audio.load_audio(audio_path)
        return self.audio
    
    def get_call_audio_box(self, n: int):
        box, power, score, accept = self.get_call(n)
        start_time, min_freq, width_time, height_freq = box

        return self.get_audio().get_sample_box(start_time, min_freq* 1000, width_time, height_freq * 1000)
  
def bulk_analysis(detect_folder, audio_folder):
    ''' Calculates basic quality and boundary arrays '''
    remove_wav = lambda x : x.replace('.wav','').replace('.WAV','')
    remove_mat = lambda x : x.replace('.mat','')

    audio_files = [audio_file.name for audio_file in os.scandir(audio_folder) if audio_file.name.endswith(('.wav','.WAV'))]
    detect_files = [detect_file.name for detect_file in os.scandir(detect_folder) if detect_file.name.endswith(('.mat'))]

    joined = []
    for a_f in audio_files:
      for d_f in detect_files:
        if d_f.startswith(remove_wav(a_f)):
          joined.append({
              'audio': a_f,
              'detect': d_f
          })
    
    all_scores = []
    all_power = []
    all_boxes = []
    all_accepts = []

    for j in tqdm.tqdm(joined):
      d_f = os.path.join(detect_folder, j['detect'])
      a_f = os.path.join(audio_folder, j['audio'])
      ds = DeepSqueak(d_f, a_f)

      for call_number in range(ds.get_call_count()):
        box, power, score, accept = ds.get_call(call_number)
        all_scores.append(score)
        all_power.append(power)
        all_accepts.append(accept)
        all_boxes.append(box)
    
    return joined, (all_boxes, all_scores, all_power, all_accepts)
    
def plot_call_quality(all_boxes, all_scores, all_power, all_accepts):
    ''' Plots histograms for Score, Power and Accept stats '''
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.set_figheight(4)
    fig.set_figwidth(15)

    ax1.hist(all_scores, color='lightblue', edgecolor='black', bins=int(180/5))
    ax1.set_title('Histogram of Scores')
    ax1.set_xlabel('Score')
    ax1.set_ylabel('Detections')

    ax2.hist(all_power, color='lightblue', edgecolor='black', bins=int(180/5))
    ax2.set_title('Histogram of Scores')
    ax2.set_xlabel('Power')
    ax2.set_ylabel('Detections')

    ax3.hist(all_accepts, color='lightblue', edgecolor='black', bins=int(180/5))
    ax3.set_title('Histogram of Accept')
    ax3.set_xlabel('Accept int(bool)')
    ax3.set_ylabel('Detections')
    
def plot_call_boundary(all_boxes, all_scores, all_power, all_accepts):
    ''' Plots histograms for the call boundary boxes '''
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
    fig.set_figheight(10)
    fig.set_figwidth(14)

    ax1.hist([b[0] for b in all_boxes], color='lightblue', edgecolor='black', bins=int(180/5))
    ax1.set_title('Call Start Time')
    ax1.set_xlabel('Offset (seconds) from clip start')
    ax1.set_ylabel('Detections')

    ax2.hist([b[0] + b[2] for b in all_boxes], color='lightblue', edgecolor='black', bins=int(180/5))
    ax2.set_title('Call End Time')
    ax2.set_xlabel('Duration')
    ax2.set_ylabel('Detections')

    ax3.hist([b[2] for b in all_boxes], color='lightblue', edgecolor='black', bins=int(180/5))
    ax3.set_title('Call Length')
    ax3.set_xlabel('Duration')
    ax3.set_ylabel('Detections')

    ax4.hist([b[1] for b in all_boxes], color='lightblue', edgecolor='black', bins=int(180/5))
    ax4.set_title('Call Low Frequency')
    ax4.set_xlabel('Frequency (kHz)')
    ax4.set_ylabel('Detections')

    ax5.hist([b[1] + b[3] for b in all_boxes], color='lightblue', edgecolor='black', bins=int(180/5))
    ax5.set_title('Call High Frequency')
    ax5.set_xlabel('Frequency (kHz)')
    ax5.set_ylabel('Detections')

    ax6.hist([b[3] for b in all_boxes], color='lightblue', edgecolor='black', bins=int(180/5))
    ax6.set_title('Call Height')
    ax6.set_xlabel('Frequency (kHz)')
    ax6.set_ylabel('Detections')
