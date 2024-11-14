import numpy as np
import torch


class StereoMatchingNetwork(torch.nn.Module):
    """
    The network should consist of the following layers:
    - Conv2d(..., out_channels=64, kernel_size=3)
    - ReLU()
    - Conv2d(..., out_channels=64, kernel_size=3)
    - ReLU()
    - Conv2d(..., out_channels=64, kernel_size=3)
    - ReLU()
    - Conv2d(..., out_channels=64, kernel_size=3)
    - functional.normalize(..., dim=1, p=2)

    Remark: Note that the convolutional layers expect the data to have shape
        `batch size * channels * height * width`. Permute the input dimensions
        accordingly for the convolutions and remember to revert it before returning the features.
    """

    def __init__(self):
        """
        Implementation of the network architecture.
        Layer output tensor size: (batch_size, n_features, height - 8, width - 8)
        """

        super().__init__()
        gpu = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=3)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(64, 64, kernel_size=3)
        self.relu2 = torch.nn.ReLU()
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3)
        self.relu3 = torch.nn.ReLU()
        self.conv4 = torch.nn.Conv2d(64, 64, kernel_size=3)
        # self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=3)
        # self.relu2 = torch.nn.ReLU()
        # self.conv3 = torch.nn.Conv2d(128, 256, kernel_size=3)
        # self.relu3 = torch.nn.ReLU()
        # self.conv4 = torch.nn.Conv2d(256, 64, kernel_size=3)
        self.normalize = torch.nn.functional.normalize

        #######################################
        # -------------------------------------
        # TODO: ENTER CODE HERE (EXERCISE 5)
        # -------------------------------------

    def forward(self, X):
        """
        The forward pass of the network. Returns the features for a given image patch.

        Args:
            X (torch.Tensor): image patch of shape (batch_size, height, width, n_channels)

        Returns:
            features (torch.Tensor): predicted normalized features of the input image patch X,
                               shape (batch_size, height - 8, width - 8, n_features)
        """
        X = self.relu1(self.conv1(X))
        X = self.relu2(self.conv2(X))
        X = self.relu3(self.conv3(X))
        X = self.relu3(self.conv4(X))
        X = self.normalize(X, p=2, dim=1)
        return X
        #######################################
        # -------------------------------------
        # TODO: ENTER CODE HERE (EXERCISE 5)
        # -------------------------------------


def calculate_similarity_score(infer_similarity_metric, Xl, Xr):
    """
    Computes the similarity score for two stereo image patches.

    Args:
        infer_similarity_metric (torch.nn.Module):  pytorch module object
        Xl (torch.Tensor): tensor holding the left image patch
        Xr (torch.Tensor): tensor holding the right image patch

    Returns:
        score (torch.Tensor): the similarity score of both image patches which is the dot product of their features
    """


    # breakpoint()
    # [128, 1, 9, 9]

    features_left = infer_similarity_metric(Xl)
    features_right = infer_similarity_metric(Xr)
    # breakpoint()
    # torch.sqrt(torch.sum(features_left ** 2, dim=1)).squeeze()
    # torch.Size([128, 64, 1, 1])

    return torch.sum(features_left * features_right, dim=1).squeeze()
    #######################################
    # -------------------------------------
    # TODO: ENTER CODE HERE (EXERCISE 5)
    # -------------------------------------

if __name__ == '__main__':
    # Test the network
    network = StereoMatchingNetwork()
    X = torch.randn(2, 1, 9, 9)
    features = network(X)
    assert features.shape[-3:] == (64, 1, 1), f"Expected shape (b, 1, 1, x), got {features.shape}"
    print("Network test successful")
    # Test the similarity score
    Xl = torch.randn(1, 1, 9, 9)
    Xr = torch.randn(1, 1, 9, 9)
    score = calculate_similarity_score(network, Xl, Xr)
    print(score,score.shape)
    # assert score.shape == (1,), f"Expected shape (1,), got {score.shape}"
    print("Similarity score test successful")