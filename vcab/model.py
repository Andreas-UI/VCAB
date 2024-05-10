import pkg_resources

import torch

from .prediction import Prediction
from .label import NUM_CLASSES

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Model:
    __model_name = "x3d_m"
    __weights = torch.load(
        pkg_resources.resource_filename("vcab", "x3d_m.ckpt"))

    def __init__(self) -> None:
        self.__model = torch.hub.load(
            "facebookresearch/pytorchvideo:main", model=self.__model_name, pretrained=True)
        self.__model.blocks[5].proj = torch.nn.Linear(
            in_features=2048, out_features=NUM_CLASSES)
        self.__model.to(device=device)

        self.__state_dict = self.__model.state_dict()
        for key in self.__weights["state_dict"].keys():
            self.__state_dict[key[6:]] = self.__weights["state_dict"][key]

        self.__model.load_state_dict(self.__state_dict)

    def predict(self, video_path: str):
        """
        Args: 
            video_path <- path to video
        Output:
            A list of tuples [(category, probability), ...]
        """
        return Prediction(model=self.__model).predict(video_path=video_path)

    def predict_stream(self, video_path: str):
        """
        Args: 
            video_path <- path to video
        Output:
            A list of list with tuples [[start_sec, end_sec, (category, probability)], ...]
        """
        return Prediction(model=self.__model).predict_stream(video_path=video_path)

    def predict_stream_emotion(self, video_path: str):
        """
        Args: 
            video_path <- path to video
        Output:
            A list of list with tuples [[start_sec, end_sec, (category, probability)], ...]
        """
        return Prediction(model=self.__model).predict_stream_emotion(video_path=video_path)
