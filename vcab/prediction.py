import torch
from .transform import Transform
from .label import NUM_CLASSES, KEY_LABEL

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Prediction:
    def __init__(self, model) -> None:
        self.__model = model
        self.__labels = KEY_LABEL

    def predict(self, video_path: str):
        """
        Predict a single video
        Args:
          video_path: path to the video file
        Returns:
          output_class_names: list of tuples (class_name, probability)
        """
        # Transform the video
        video_data = Transform().transform(video_path=video_path)

        # Push video data to cpu/gpu device
        video = video_data["video"]
        video = video.to(device)

        act = torch.nn.Softmax(dim=1)

        # Predict the video action category
        output = self.__model(video[None, ...])
        output = act(output)
        output_classes = output.topk(k=NUM_CLASSES).indices[0]

        # Return the actions and the probabilities
        output_class_names = [(self.__labels[int(i)], round(
            output[0][int(i)].item() * 100, 2)) for i in output_classes]

        return output_class_names

    def predict_stream(self, video_path: str):
        """
        Predict a stream of video
        Args:
          video_path: path to the video file
        Returns:
          output_predictions: list of tuples (start_sec, end_sec, output_class_names)
        """
        # Stream transform the video
        video_datas = Transform().transform_stream(video_path=video_path)

        # Loop through the timestamp in the stream data
        output_predictions = []
        for start_sec, end_sec, video_data in video_datas:
            # Push video data to cpu/gpu device
            video = video_data["video"]
            video = video.to(device)

            act = torch.nn.Softmax(dim=1)

            # Predict the video action category
            output = self.__model(video[None, ...])
            output = act(output)
            output_classes = output.topk(k=NUM_CLASSES).indices[0]

            # Append the timestamp and the output of the category
            output_class_names = [(self.__labels[int(i)], round(
                output[0][int(i)].item() * 100, 2)) for i in output_classes]
            output_predictions.append((start_sec, end_sec, output_class_names))

        return output_predictions
