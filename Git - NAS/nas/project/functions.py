import os
import shutil
import pickle
import numpy as np
from itertools import groupby
from matplotlib import pyplot as plt

from params import *
from MLP import MLPSearchSpace


########################################################
#                   DATA PROCESSING                    #
########################################################


def unison_shuffled_copies(a, b):
    """
    Description
    ---------------
    Shuffle lists a and b randomly, in the same order
    (if a[1] becomes a[5], then, b[1] becomes b[5]).
    Input(s)
    ---------------
    a: array
    b: array
    Output(s)
    ---------------
    a[p]: array
    b[p]: array
    """
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


########################################################
#                       LOGGING                        #
########################################################

def clean_log(path_to_use):
    """
    Description
    ---------------
    This function removes all existing logs of previous models generated
    """
    filelist = os.listdir(path_to_use)
    for file in filelist:
        if os.path.isfile(path_to_use + '/{}'.format(file)):
            os.remove(path_to_use + '/{}'.format(file))


def log_event():
    dest = 'LOGS'
    while os.path.exists(dest):
        dest = 'LOGS/event{}'.format(np.random.randint(10000))
    os.mkdir(dest)
    filelist = os.listdir('LOGS')
    for file in filelist:
        if os.path.isfile('LOGS/{}'.format(file)):
            shutil.move('LOGS/{}'.format(file),dest)


def get_latest_event_id():
    all_subdirs = ['LOGS/' + d for d in os.listdir('LOGS') if os.path.isdir('LOGS/' + d)]
    latest_subdir = max(all_subdirs, key=os.path.getmtime)
    return int(latest_subdir.replace('LOGS/event', ''))


########################################################
#                 RESULTS PROCESSING                   #
########################################################


""" INITIAL VERSION
def load_nas_data():
    event = get_latest_event_id()
    data_file = 'LOGS/event{}/nas_data.pkl'.format(event)
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    return data """


def load_nas_data(number = None):
    """
    Description
    ---------------
    Loads the data about the models.
    Output(s)
    ---------------
    data: array
    """
    if number == None:
        data_file = './LOGS/nas_data.pkl'
    else:
        data_file = './LOGS/event{}/nas_data.pkl'.format(number)
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    return data


def sort_search_data(nas_data):
    """
    Description
    ---------------
    This function sorts the data by their value accuracy.
    Input(s)
    ---------------
    nas_data: array
    Output(s)
    ---------------
    nas_data: array
    """
    val_accs = [item[1] for item in nas_data]
    sorted_idx = np.argsort(val_accs)[::-1]
    nas_data = [nas_data[x] for x in sorted_idx]
    return nas_data

########################################################
#                EVALUATION AND PLOTS                  #
########################################################

def get_top_n_architectures(n):
    data = load_nas_data()
    data = sort_search_data(data)
    search_space = MLPSearchSpace(TARGET_CLASSES)
    print('Top {} Architectures:'.format(n))
    for seq_data in data[:n]:
        print('Architecture', search_space.decode_sequence(seq_data[0]))
        print('Validation Accuracy:', seq_data[1])


def get_nas_accuracy_plot():
    data = load_nas_data()
    accuracies = [x[1] for x in data]
    plt.plot(np.arange(len(data)), accuracies)
    plt.show()


def get_accuracy_distribution():
    event = get_latest_event_id()
    data = load_nas_data()
    accuracies = [x[1]*100. for x in data]
    accuracies = [int(x) for x in accuracies]
    sorted_accs = np.sort(accuracies)
    count_dict = {k: len(list(v)) for k, v in groupby(sorted_accs)}
    plt.bar(list(count_dict.keys()), list(count_dict.values()))
    plt.show()



