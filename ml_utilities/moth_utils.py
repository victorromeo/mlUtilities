import os, sys, datetime, time, dateutil, shutil
from dateutil.tz import tzutc, tzlocal

def timestamp_to_datetime_local(timestamp: int):
    local_tz = datetime.datetime.now().astimezone().tzinfo
    return datetime.datetime.fromtimestamp(timestamp, tz = local_tz)

def timestamp_to_datetime_utc(timestamp: int):
    return datetime.datetime.utcfromtimestamp(timestamp)

def parse_filename_to_timestamp(filename, default):
    try:
        basename = os.path.basename(filename).replace('.WAV','').replace('.wav','')
        if '_' in basename:
            return datetime.datetime.strptime(basename,'%Y%m%d_%H%M%S').timestamp()
        return int(basename, 16)
    except Exception as e: 
        print(f'Unable to determine timestamp: {filename}. {e}')
        return default

def parse_filename_to_datetime_local(filename, default):
    timestamp = parse_filename_to_timestamp(filename, None)
    try:
        return timestamp_to_datetime_local(timestamp)
    except Exception as e:
        print(f'Unable to determine date {filename}. {e}')
        return default

def parse_filename_to_datetime_utc(filename, default):
    timestamp = parse_filename_to_timestamp(filename, None)
    try:
        return timestamp_to_datetime_utc(timestamp)
    except:
        return default

def timestamp_to_hex_string(timestamp):
    return format(timestamp, 'x').upper()

def datetime_to_datetime_string(value):
    return f'{value:%Y%m%d_%H%M%S}'

def rename_files_timestamp_to_local(src_path, dst_path):
    src_items = os.scandir(src_path)
    for src_item in src_items:
        value_local = parse_filename_to_datetime_local(src_item.name, None)
        
        if value_local is not None:
            dst_name = datetime_to_datetime_string(value_local)
            shutil.move(src_item.path, os.path.join(dst_path, f'{dst_name}.wav'))
    
def rename_files_timestamp_to_utc(src_path, dst_path):
    src_items = os.scandir(src_path)

    for src_item in src_items:
        value_utc = parse_filename_to_datetime_utc(src_item.name, None)
        
        if value_utc is not None:
            dst_name = datetime_to_datetime_string(value_utc)
            shutil.move(src_item.path, os.path.join(dst_path, f'{dst_name}.wav'))

def utc_to_local(utc_value):
    epoch = time.mktime(utc_value.timetuple())
    offset = datetime.datetime.fromtimestamp(epoch) - datetime.datetime.utcfromtimestamp(epoch)
    return utc_value + offset

def rename_files_utc_to_local(src_path, dst_path):
    src_items = os.scandir(src_path)

    for src_item in src_items:
        value_utc = parse_filename_to_datetime_utc(src_item.name, None)

        if value_utc is not None:
            dst_name = datetime_to_datetime_string(utc_to_local(value_utc))
            shutil.move(src_item.path, os.path.join(dst_path, f'{dst_name}.wav'))

def rename_files_local_to_utc(src_path, dst_path):
    src_items = os.scandir(src_path)
    for src_item in src_items:
        value_local = parse_filename_to_datetime_local(src_item.name, None)

        if value_local is not None:
            dst_name = datetime_to_datetime_string(value_local.astimezone(tzutc()))
            shutil.move(src_item.path, os.path.join(dst_path, f'{dst_name}.wav'))

if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser('Conversion of filenames for audio recordings made by AudioMoth')

    ap.add_argument('-s','--source', required=True, help='source path')
    ap.add_argument('-d', '--dest', required=True, help='destination path')

    mode_group = ap.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--utc_to_local', help='Convert UTC dates to Local dates', action='store_true')
    mode_group.add_argument('--local_to_utc', help='Convert UTC dates to Local dates', action='store_true')
    mode_group.add_argument('--timestamp_to_local',  help='Convert HEX timestamp files to local date files', action='store_true')
    mode_group.add_argument('--timestamp_to_utc', help='Convert HEX timestamp files to UTC date files', action='store_true')
    args = vars(ap.parse_args())

    operations = {
        'utc_to_local' : lambda: rename_files_utc_to_local(args['source'],args['dest']),
        'local_to_utc' : lambda: rename_files_local_to_utc(args['source'],args['dest']),
        'timestamp_to_local' : lambda: rename_files_timestamp_to_local(args['source'],args['dest']),
        'timestamp_to_utc' : lambda: rename_files_timestamp_to_utc(args['source'],args['dest']),
    }

    for key in operations.keys():
        if args[key]:
            operations[key]()