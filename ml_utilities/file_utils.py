import os, subprocess, platform

wav_extensions = ('.wav','.WAV')
mat_extensions = ('.mat')

def get_files(path, extensions = None):
    ''' Used to safetly fetch file names from a directory, as os.scandir is more likely to fail if directory large'''

    if platform.system() not in ['Linux', 'Darwin']:
        # Windows system so use standard os.walk
        file_list = []
        for root, dirs, files in os.walk(path, topdown=False):
            abs_path = os.path.realpath(root)
            file_list = file_list + (list([os.path.join(abs_path, i) for i in files if i.endswith(extensions)]) if extensions is not None else list([os.path.join(abs_path, i) for i in files]))
        
        return file_list
    else:
        # OS X or Linux system so use 'find' posix operation
        abs_path = os.path.realpath(path)
        if extensions is not None:
            if type(extensions) is list or type(extensions) is tuple:
                grep = '|'.join(extensions)
                cmd = f"find '{abs_path}' -type f | grep -E '{grep}'"
                print(cmd)
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
            else:
                grep = extensions
                process = subprocess.Popen(f"find '{abs_path}' -type f | grep '{grep}'", stdout=subprocess.PIPE, shell=True)
        else:
            process = subprocess.Popen(f"find '{abs_path}' -type f", stdout=subprocess.PIPE, shell=True)
            
        stdout = process.communicate()[0]

        files = stdout.decode('utf-8').split('\n')
        files = sorted([f for f in files if f is not None and len(f) > 0])

        return files

def get_mat_files(path):
    ''' Get Matlab mat files '''
    return get_files(path, mat_extensions)

def get_wav_files(path):
    ''' Get Audio WAV files '''
    return get_files(path, wav_extensions)