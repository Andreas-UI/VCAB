from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import CenterCropVideo, NormalizeVideo

from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample
from pytorchvideo.data.encoded_video import EncodedVideo


class Transform:
    __mean = [0.45, 0.45, 0.45]
    __std = [0.225, 0.225, 0.225]
    __frames_per_second = 30

    __model_name = "x3d_m"
    __model_transform_params = {
        "x3d_xs": {
            "side_size": 182,
            "crop_size": 182,
            "num_frames": 4,
            "sampling_rate": 12,
        },
        "x3d_s": {
            "side_size": 182,
            "crop_size": 182,
            "num_frames": 13,
            "sampling_rate": 6,
        },
        "x3d_m": {
            "side_size": 256,
            "crop_size": 256,
            "num_frames": 16,
            "sampling_rate": 5,
        }
    }

    def __init__(self) -> None:
        pass

    def transform(self, video_path: str):
        """
        Transform a single video
        Args:
          video_path: path to the video file
        Returns:
          video_data: transformed video data
        """
        transform_params = self.__model_transform_params[self.__model_name]
        clip_duration = (
            transform_params["num_frames"] * transform_params["sampling_rate"]) / self.__frames_per_second
        start_sec = 0
        end_sec = start_sec + clip_duration

        transform = Compose([
            ApplyTransformToKey(
                key="video",
                transform=Compose([
                    UniformTemporalSubsample(transform_params["num_frames"]),
                    Lambda(lambda x: x/255.0),
                    NormalizeVideo(self.__mean, self.__std),
                    ShortSideScale(size=transform_params["side_size"]),
                    CenterCropVideo(crop_size=(
                        transform_params["crop_size"], transform_params["crop_size"]))
                ]),
            ),
        ])

        video = EncodedVideo.from_path(video_path)
        video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)

        return transform(video_data)

    def transform_stream(self, video_path: str):
        """
        Transform a stream of video
        Args:
          video_path: path to the video file
        Returns:
          video_data: transformed streams of video data with timestamp
        """
        transform_params = self.__model_transform_params[self.__model_name]
        clip_duration = (
            transform_params["num_frames"] * transform_params["sampling_rate"]) / self.__frames_per_second
        start_sec = 0
        end_sec = start_sec + clip_duration

        transform = Compose([
            ApplyTransformToKey(
                key="video",
                transform=Compose([
                    UniformTemporalSubsample(transform_params["num_frames"]),
                    Lambda(lambda x: x/255.0),
                    NormalizeVideo(self.__mean, self.__std),
                    ShortSideScale(size=transform_params["side_size"]),
                    CenterCropVideo(crop_size=(
                        transform_params["crop_size"], transform_params["crop_size"]))
                ]),
            ),
        ])

        video = EncodedVideo.from_path(video_path)
        duration = float(video.duration)

        transformed_data = []
        while end_sec < duration:
            video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)
            transformed_data.append(
                (start_sec, end_sec, transform(video_data)))

            start_sec = end_sec
            end_sec += clip_duration

        # get the last uncatch frames
        video_data = video.get_clip(start_sec=start_sec, end_sec=duration)
        transformed_data.append((start_sec, end_sec, transform(video_data)))

        return transformed_data
