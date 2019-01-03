import config


if __name__ == '__main__':
    # Loads the arguments from the config file and command line and sets the description for the application.
    arguments = config.load_config(description="Active Learning Experiment Framework")
