import config
import dataset

import os
import torch
import numpy as np


def log(arguments, message):
    """
    Function to handle printing and logging od messages.
    :param arguments: An ArgumentParser object.
    :param message: String with the message to be printed or logged.
    """

    if arguments.verbose:
        print(message)
    if arguments.log_file != '':
        if not os.path.isfile(arguments.log_file):
            os.makedirs(os.path.dirname(arguments.log_file))
        print(message, file=open(arguments.log_file, 'a'))


if __name__ == '__main__':
    # Loads the arguments from the config file and command line and sets the description for the application.
    arguments = config.load_config(description="Active Learning Experiment Framework")

    log(arguments, "Arguments Loaded")

    # Sets the seeds for numpy and pytorch to the defined seeds.
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)

    # Sets CUDNN to be used by torch,
    torch.backends.cudnn.enabled = True

    # Extracts the training and testing data from the defined dataset.
    x_train, y_train, x_test, y_test = dataset.get_dataset()

    """
    To conduct Active Learning a binary array is used that states if a peice of data should used within the labeled
    training data or should be treated as unlabeled data. An initial random sample of data of a specified size if then
    marked as labelled. 
    """
    labeled_indices = np.zeros(len(y_train), dtype=bool)
    labeled_temp = np.arange(len(y_train))
    np.random.shuffle(labeled_temp)
    labeled_indices[labeled_temp[:arguments.init_labels]] = True
