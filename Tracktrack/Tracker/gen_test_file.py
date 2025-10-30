import os
import shutil

trackers = ['mot17_test', 'mot20_test']


for tracker in trackers:
    path = '../outputs/3. track/' + tracker + '/'
    files = os.listdir(path)

    if 'mot17' in tracker:
        for file in files:
            file_path = path + file
            shutil.copy(file_path, file_path.replace('FRCNN', 'SDP'))
            shutil.copy(file_path, file_path.replace('FRCNN', 'DPM'))

        dummy_path = './utils/mot17_dummy/'
        dummy_files = os.listdir(dummy_path)
        for dummy_file in dummy_files:
            shutil.copy(dummy_path + dummy_file, path + dummy_file)

    if 'mot20' in tracker:
        dummy_path = './utils/mot20_dummy/'
        dummy_files = os.listdir(dummy_path)
        for dummy_file in dummy_files:
            shutil.copy(dummy_path + dummy_file, path.replace('mot17', 'mot20') + dummy_file)
