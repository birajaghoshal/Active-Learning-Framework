import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset
from imgaug import augmenters as iaa


def get_dataset():
    """
    Function for loading the dataset and returning four arrays containing:
     training data, training labels, testing data and testing labels.
    This function loads the MNIST dataset using a function from the Torch library.
    :return: Four Arrays containing training data, training labels, testing data and testing labels.
    """

    training = datasets.MNIST("./MNIST", train=True, download=True)
    testing = datasets.MNIST("./MNIST", train=False, download=True)
    x_train = training.train_data
    y_train = training.train_labels
    x_test = testing.test_data
    y_test = testing.test_labels
    return x_train, y_train, x_test, y_test


def transform():
    """
    Function that will be used within the data handler to transform the input images.
    This can be modified to adjust the transforms the images to the specific dataset.
    If no transforms is required this function can return None.
    :return: A transform function.
    """
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])


class DataHandler(Dataset):
    """
    Class for handling the dataset when being used to train a neural network.
    """

    def __init__(self, x, y, augmentation=False):
        """
        Initilisation method for the data handler class that sets the initial parameters.
        :param x: An array of data.
        :param y: An array of labels.
        """

        if augmentation:
            a = iaa.Sequential([iaa.Fliplr(1.0)]).augment_images(x)
            b = iaa.Sequential([iaa.Flipud(1.0)]).augment_images(x)
            c = iaa.Sequential([iaa.GaussianBlur(1.0)]).augment_images(x)
            d = iaa.Sequential([iaa.ChannelShuffle(1.0)]).augment_images(x)
            e = iaa.Sequential([iaa.Sharpen(1.0)]).augment_images(x)

            self.x = np.concatenate([x, a, b, c, d, e])
            self.y = np.concatenate([y, y, y, y, y, y])
        self.transform = transform()

    def __getitem__(self, index):
        """
        Method to get a single item from the dataset from a specific index.
        :param index: The location in the array where the data will be taken from.
        :return: A single item of data, the label for the data and the index where it was extracted from.
        """

        x, y = self.x[index], self.y[index]
        if transforms is not None:
            x = Image.fromarray(x.numpy(), mode='L')
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        """
        Method for returning the size of dataset handled by the data handler.
        :return: An integer containing the length of the dataset.
        """

        return len(self.x)
