import os

wav_extensions = ('.wav','.WAV')
mat_extensions = ('.mat')

def get_files(path, extensions = None):
  ''' Used to safetly fetch file names from a directory, as os.scandir is more likely to fail if directory large'''
  file_list = []
  for root, dirs, files in os.walk(path, topdown=False):
    abs_path = os.path.realpath(root)
    file_list = file_list + (list([os.path.join(abs_path, i) for i in files if i.endswith(extensions)]) if extensions is not None else list([os.path.join(abs_path, i) for i in files]))
  
  return file_list

def get_mat_files(path):
    ''' Get Matlab mat files '''
    return get_files(path, mat_extensions)

def get_wav_files(path):
    ''' Get Audio WAV files '''
    return get_files(path, wav_extensions)