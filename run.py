import model
import config
import dataset
import strategy

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
    data_handler = dataset.DataHandler

    """
    To conduct Active Learning a binary array is used that states if a peice of data should used within the labeled
    training data or should be treated as unlabeled data. An initial random sample of data of a specified size if then
    marked as labelled. 
    """
    labeled_indices = np.zeros(len(y_train), dtype=bool)
    labeled_temp = np.arange(len(y_train))
    np.random.shuffle(labeled_temp)
    labeled_indices[labeled_temp[:arguments.init_labels]] = True

    """
    This sets the model that will be used within the active learning experiments this can be replaced by any other
    pytorch model based of the nn.module class. The forward pass method also needs to include two outputs, the output
    of the model and an intermediate output from the centre (if not needed it can return None as the second output).
    Another method needed to be implemented is the get_embedding_dim that should specify the size of the
    intermediate output from the model.
    """
    model = model.Model()

    query_strategy = None

    # TODO QUERY STRATEGIES GO HERE
    # TODO COMMENT ALL CODE BELOW HERE

    if query_strategy is None:
        query_strategy = strategy.Strategy(x_train, y_train, labeled_indices, model, data_handler, arguments)
        labeled_indices[:] = True
        arguments.num_iterations = 0

    log(arguments, "\nNumber of initial labeled data: {}".format(list(labeled_indices).count(True)))
    log(arguments, "Number of initial unlabeled data: {}".format(len(y_train) - list(labeled_indices).count(True)))
    log(arguments, "Number of testing data: {}".format(len(y_test)))

    log(arguments, "\n---------- Iteration 0")
    query_strategy.train()
    _, predictions = query_strategy.predict(x_test, y_test)
    accuracy = np.zeros(arguments.num_iterations + 1)
    accuracy[0] = 1.0 * (y_test == predictions).sum().item() / len(y_test)
    log(arguments, "\nTesting Accuracy {}\n\n\n".format(accuracy[0]))

    for iteration in range(1, arguments.num_iterations+1):
        log(arguments, "\n---------- Iteration {}".format(iteration))

        query_indices = query_strategy.query(arguments.query_labels)
        labeled_indices[query_indices] = True
        query_strategy.update(labeled_indices)
        query_strategy.train()

        _, predictions = query_strategy.predict(x_test, y_test)
        accuracy[iteration] = 1.0 * (y_test == predictions).sum().item() / len(y_test)
        log(arguments, "Testing Accuracy {}\n\n\n".format(accuracy[iteration]))

    log(arguments, accuracy)
