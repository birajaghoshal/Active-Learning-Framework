import config

import os


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
