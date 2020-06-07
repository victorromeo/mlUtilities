import os, keras
from keras.datasets import mnist, cifar, cifar10, cifar100, fashion_mnist

# Audio

## LibriSpeech ASR Corpus http://www.openslr.org/12/
## VoxCeleb http://www.robots.ox.ac.uk/~vgg/data/voxceleb
## Speech Commands https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html
## Free Sound Database https://datasets.freesound.org/fsd/
## Urban Sound Dataset https://urbansounddataset.weebly.com/urbansound.html
## Bird Audio Detection challenge http://machine-listening.eecs.qmul.ac.uk/bird-audio-detection-challenge/
## ISOLET Data Set challenge https://data.world/uci/isolet
## Zero Resource Speech challenge https://github.com/bootphon/zerospeech2017
## Chime https://archive.org/details/chime-home
## Common Voice https://www.kaggle.com/mozillaorg/common-voice/home
## TED-LIUM audio transcriptions http://www.openslr.org/51/
## Google AudioSet https://research.google.com/audioset/

## Tensorflow dataset  https://www.tensorflow.org/datasets/overview#find_available_datasets
tensorflow_datasets = ['abstract_reasoning',
    'aeslc',
    'aflw2k3d',
    'amazon_us_reviews',
    'arc',
    'bair_robot_pushing_small',
    'beans',
    'big_patent',
    'bigearthnet',
    'billsum',
    'binarized_mnist',
    'binary_alpha_digits',
    'blimp',
    'c4',
    'caltech101',
    'caltech_birds2010',
    'caltech_birds2011',
    'cars196',
    'cassava',
    'cats_vs_dogs',
    'celeb_a',
    'celeb_a_hq',
    'cfq',
    'chexpert',
    'cifar10',
    'cifar100',
    'cifar10_1',
    'cifar10_corrupted',
    'citrus_leaves',
    'cityscapes',
    'civil_comments',
    'clevr',
    'cmaterdb',
    'cnn_dailymail',
    'coco',
    'coil100',
    'colorectal_histology',
    'colorectal_histology_large',
    'common_voice',
    'cos_e',
    'crema_d',
    'curated_breast_imaging_ddsm',
    'cycle_gan',
    'deep_weeds',
    'definite_pronoun_resolution',
    'dementiabank',
    'diabetic_retinopathy_detection',
    'div2k',
    'dmlab',
    'downsampled_imagenet',
    'dsprites',
    'dtd',
    'duke_ultrasound',
    'emnist',
    'eraser_multi_rc',
    'esnli',
    'eurosat',
    'fashion_mnist',
    'flic',
    'flores',
    'food101',
    'forest_fires',
    'gap',
    'geirhos_conflict_stimuli',
    'german_credit_numeric',
    'gigaword',
    'glue',
    'groove',
    'higgs',
    'horses_or_humans',
    'i_naturalist2017',
    'image_label_folder',
    'imagenet2012',
    'imagenet2012_corrupted',
    'imagenet2012_subset',
    'imagenet_resized',
    'imagenette',
    'imagewang',
    'imdb_reviews',
    'iris',
    'kitti',
    'kmnist',
    'lfw',
    'librispeech',
    'librispeech_lm',
    'libritts',
    'ljspeech',
    'lm1b',
    'lost_and_found',
    'lsun',
    'malaria',
    'math_dataset',
    'mnist',
    'mnist_corrupted',
    'movie_rationales',
    'moving_mnist',
    'multi_news',
    'multi_nli',
    'multi_nli_mismatch',
    'natural_questions',
    'newsroom',
    'nsynth',
    'omniglot',
    'open_images_challenge2019_detection',
    'open_images_v4',
    'opinosis',
    'oxford_flowers102',
    'oxford_iiit_pet',
    'para_crawl',
    'patch_camelyon',
    'pet_finder',
    'places365_small',
    'plant_leaves',
    'plant_village',
    'plantae_k',
    'qa4mre',
    'quickdraw_bitmap',
    'reddit',
    'reddit_tifu',
    'resisc45',
    'robonet',
    'rock_paper_scissors',
    'rock_you',
    'samsum',
    'savee',
    'scan',
    'scene_parse150',
    'scicite',
    'scientific_papers',
    'shapes3d',
    'smallnorb',
    'snli',
    'so2sat',
    'speech_commands',
    'squad',
    'stanford_dogs',
    'stanford_online_products',
    'starcraft_video',
    'stl10',
    'sun397',
    'super_glue',
    'svhn_cropped',
    'ted_hrlr_translate',
    'ted_multi_translate',
    'tedlium',
    'tf_flowers',
    'the300w_lp',
    'tiny_shakespeare',
    'titanic',
    'trivia_qa',
    'uc_merced',
    'ucf101',
    'vgg_face2',
    'visual_domain_decathlon',
    'voc',
    'voxceleb',
    'waymo_open_dataset',
    'web_questions',
    'wider_face',
    'wiki40b',
    'wikihow',
    'wikipedia',
    'wmt14_translate',
    'wmt15_translate',
    'wmt16_translate',
    'wmt17_translate',
    'wmt18_translate',
    'wmt19_translate',
    'wmt_t2t_translate',
    'wmt_translate',
    'xnli',
    'xsum',
    'yelp_polarity_reviews']

## Get Standard Tensorflow dataset
def load_tfds(name:str=None, split:str = None):
    assert name is not None and len(name) > 1, 'Argument name is required'
    assert name in tensorflow_datasets, f'Unknown dataset name: {name}'
    
    # Install prerequisites
    os.system('pip install --quiet tensorflow-datasets')
    import tensorflow_datasets as tfds
    import tensorflow as tf

    # Construct a tf.data.Dataset
    return tfds.load(name, split=split, shuffle_files=True)

def load_tfds_dataset_builder(name:str=None, split:str = None, download:bool = True, prepare:bool = True, path:str = None):
    assert name is not None and len(name) > 1, 'Argument name is required'
    assert name in tensorflow_datasets, f'Unknown dataset name: {name}'
    
    # Install prerequisites
    os.system('pip install --quiet tensorflow-datasets')
    import tensorflow_datasets as tfds
    import tensorflow as tf

    if path is None or len(path) == 0:
        path = name

    dsbuilder = tfds.builder(name)

    if download:
        dsbuilder.download(download_dir = path)
    
    if prepare:
        dsbuilder.prepare()

    return dsbuilder

## Visual datasets

def load_mnist():
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    return (x_train, y_train), (x_test, y_test)

def load_cifar10():
    from keras.datasets import cifar, cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    return (x_train, y_train), (x_test, y_test)

def load_cifar100():
    from keras.datasets import cifar, cifar100
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()

    return (x_train, y_train), (x_test, y_test)

## Audio datasets

def load_esc50(path='esc50'):
    import csv
    from ml_utilities import ESC50
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=False)
        os.system(f"git clone --quiet -s https://github.com/karolpiczak/ESC-50.git {path}")

    return ESC50(path)

def load_fsdd(path='fsdd'):
    import csv
    from ml_utilities import ESC50
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=False)
        os.system(f"git clone --quiet -s https://github.com/Jakobovski/free-spoken-digit-dataset.git {path}")
