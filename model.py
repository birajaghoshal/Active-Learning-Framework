import torch.nn as nn
import torch.nn.functional as functional


class Model(nn.Module):
    """
    The class for the model that will be used within the active learning experiments.
    This class can be replaced with a custom model depending on the dataset being used
    or task for the model such as segmentation.
    """

    def __init__(self):
        """
        The initialiser for the Model class that sets the functions of the models.
        """

        # Call the initiliser for the parent class.
        super(Model, self).__init__()

        # Defines the functions for each layer in the neural network.
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        """
        This method for the model is used to perform a forward pass.
        This method returns the output from the model and an intermediate output from the model
        used in some active learning query strategy.
        :param x: The data that will be passed through the model.
        :return: The output of the model.
        """

        # Uses the defined functions to perform a forward pass.
        conv1 = functional.relu(functional.max_pool2d(self.conv1(x), 2))
        conv2 = functional.relu(functional.max_pool2d(self.conv2_drop(self.conv2(conv1)), 2))
        flat = conv2.view(-1, 320)
        fc1 = functional.relu(self.fc1(flat))
        out = functional.relu(self.fc2(fc1))

        # Method outputs the final output from the model and a intermediate output.
        return out, fc1
