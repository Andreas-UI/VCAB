import cv2
import ffmpegcv
import torch
from ultralytics import YOLO
import pkg_resources
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from .transform import Transform
from .label import NUM_CLASSES, KEY_LABEL
from .neuralnetwork import NeuralNetwork

from MEGraphAU.OpenGraphAU.predict import predict
from MEGraphAU.OpenGraphAU.utils import Image, draw_text

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Prediction:
    yolo = YOLO(pkg_resources.resource_filename("vcab", "yolov8n-face.pt"))

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

    def predict_stream_emotion(self, video_path: str):
        """
        Predict a stream of video with emotion from https://github.com/Andreas-UI/ME-GraphAU-Video.git
        Args:
          video_path: path to the video file
        Returns:
          action_predictions: the action predictions streamed from the video.
          emotion_predictions: the emotion predictions streamed from the video.
          autism_predictions: the autism predictions streamed from the video.
          autism_percentage: the percentage severity of autism in the video.
        """
        # Stream transform the video
        video_datas = Transform().transform_stream(video_path=video_path)

        # Loop through the timestamp in the stream data
        action_predictions = []
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
            action_predictions.append((start_sec, end_sec, output_class_names))

        # Predict the video emotions
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        emotion_predictions = {}
        output_frames = []

        while (cap.isOpened()):
            ret, frame = cap.read()

            if ret:
                frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
                current_time = frame_number / fps

                faces = self.yolo.predict(frame, conf=0.4, iou=0.3, verbose=False)

                for face in faces:
                    parameters = face.boxes

                    for box in parameters:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        h, w = y2 - y1, x2 - x1

                        _faces = frame[y1:y1 + h, x1:x1 + w]

                        infostr_aus, pred = predict(Image.fromarray(_faces))
                        res, _ = draw_text(frame, list(
                            infostr_aus), pred, ((x1, y1), (x1+w, y1+h)))
                        emotion_predictions[current_time] = res

                        frame = cv2.rectangle(
                            frame, (x1, y1), (x1+w, y1+h), (0, 0, 255), 2)

                output_frames.append(frame)

            else:
                break

        cap.release()
        output_video = ffmpegcv.VideoWriter(f"{video_path[:-4]}_emotion.mp4", None, fps)

        for of in output_frames:
            output_video.write(of)
        output_video.release()

        # Concatenate results
        output_predictions = []
        i = 0
        for action in action_predictions:
            start_time, end_time, acts = action
            label = "normal" if acts[0][0] not in [
                "headbanging", "armflapping", "spinning"] else "autism"
            acts = sorted(acts, key=lambda x: x[0])

            for time, emotion in emotion_predictions.items():
                if (float(time) >= start_time) and (float(time) <= end_time):
                    fau = [value for _, value in emotion.items()]
                    fau.extend([a[1] for a in acts])
                    fau.append(label)
                    fau.insert(0, time)
                    output_predictions.append(fau)
                    i += 1

        # Multimodal
        columns = ["Time", "AU1", "AU2", "AU4", "AU5", "AU6", "AU7", "AU9",
                   "AU10", "AU11", "AU12", "AU13", "AU14", "AU15", "AU16",
                   "AU17", "AU18", "AU19", "AU20", "AU22", "AU23", "AU24",
                   "AU25", "AU26", "AU27", "AU32", "AU38", "AU39", "AUL1",
                   "AUR1", "AUL2", "AUR2", "AUL4", "AUR4", "AUL6", "AUR6",
                   "AUL10", "AUR10", "AUL12", "AUR12", "AUL14", "AUR14",
                   "ArmFlapping", "HeadBanging", "Normal", "Spinning",
                   "Label"]

        dropped_columns = ["Time", "Label"]

        label_encoder = LabelEncoder()
        label_encoder.fit(["normal", "autism"])

        test = pd.DataFrame(output_predictions, columns=columns)
        X_test = test.drop(columns=dropped_columns, axis=1).values
        y_test = label_encoder.transform(test['Label'].values)

        test_data = TensorDataset(torch.tensor(
            X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
        test_loader = DataLoader(test_data, batch_size=32)

        input_size = X_test.shape[1]
        num_classes = len(label_encoder.classes_)

        model = NeuralNetwork(input_size, [64, 32, 16], num_classes)
        model.load_state_dict(torch.load(pkg_resources.resource_filename("vcab", "best_model.pth")))

        model.eval()
        test_logits=[]
        autism_predictions={}
        with torch.no_grad():
            for inputs, _ in test_loader:
                outputs=model(inputs)
                test_logits.extend(outputs.numpy())

        test_probs = np.exp(test_logits) / np.sum(np.exp(test_logits), axis=1, keepdims=True)

        autism_count=0
        for i, probs in enumerate(test_probs):
            # print(f"Sample {i+1}:")
            for class_idx, prob in enumerate(probs):
                # print(f"Class {label_encoder.classes_[class_idx]}: {prob * 100:.2f}%")
                if label_encoder.classes_[class_idx] == "autism":
                    time = test.at[i, 'Time']
                    if prob >= 0.5:
                        autism_count += 1
                        autism_predictions[time] = "symptoms"
                    else:
                        autism_predictions[time] = "no_symptoms"

        autism_percentage=autism_count/len(test_probs) * 100

        return action_predictions, emotion_predictions, autism_predictions, autism_percentage, f"{video_path[:-4]}_emotion.mp4"

    def predict_stream_emotion_stage_1(self, video_path: str):
        """
        Predict a stream of video with emotion from https://github.com/Andreas-UI/ME-GraphAU-Video.git
        Args:
          video_path: path to the video file
        Returns:
          action_predictions: the action predictions streamed from the video.
          emotion_predictions: the emotion predictions streamed from the video.
          autism_predictions: the autism predictions streamed from the video.
          autism_percentage: the percentage severity of autism in the video.
        """
        # Stream transform the video
        video_datas = Transform().transform_stream(video_path=video_path)

        # Loop through the timestamp in the stream data
        action_predictions = []
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
            action_predictions.append((start_sec, end_sec, output_class_names))

        # Predict the video emotions
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        emotion_predictions = {}
        output_frames = []

        while (cap.isOpened()):
            ret, frame = cap.read()

            if ret:
                frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
                current_time = frame_number / fps

                faces = self.yolo.predict(frame, conf=0.4, iou=0.3, verbose=False)

                for face in faces:
                    parameters = face.boxes

                    for box in parameters:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        h, w = y2 - y1, x2 - x1

                        _faces = frame[y1:y1 + h, x1:x1 + w]

                        infostr_aus, pred = predict(Image.fromarray(_faces))
                        res, _ = draw_text(frame, list(
                            infostr_aus), pred, ((x1, y1), (x1+w, y1+h)))
                        emotion_predictions[current_time] = res

                        frame = cv2.rectangle(
                            frame, (x1, y1), (x1+w, y1+h), (0, 0, 255), 2)

                output_frames.append(frame)

            else:
                break

        cap.release()

        # Concatenate results
        output_predictions = []
        i = 0
        for action in action_predictions:
            start_time, end_time, acts = action
            label = "normal" if acts[0][0] not in [
                "headbanging", "armflapping", "spinning"] else "autism"
            acts = sorted(acts, key=lambda x: x[0])

            for time, emotion in emotion_predictions.items():
                if (float(time) >= start_time) and (float(time) <= end_time):
                    fau = [value for _, value in emotion.items()]
                    fau.extend([a[1] for a in acts])
                    fau.append(label)
                    fau.insert(0, time)
                    output_predictions.append(fau)
                    i += 1

        # Multimodal
        columns = ["Time", "AU1", "AU2", "AU4", "AU5", "AU6", "AU7", "AU9",
                   "AU10", "AU11", "AU12", "AU13", "AU14", "AU15", "AU16",
                   "AU17", "AU18", "AU19", "AU20", "AU22", "AU23", "AU24",
                   "AU25", "AU26", "AU27", "AU32", "AU38", "AU39", "AUL1",
                   "AUR1", "AUL2", "AUR2", "AUL4", "AUR4", "AUL6", "AUR6",
                   "AUL10", "AUR10", "AUL12", "AUR12", "AUL14", "AUR14",
                   "ArmFlapping", "HeadBanging", "Normal", "Spinning",
                   "Label"]

        dataframe = pd.DataFrame(output_predictions, columns=columns)
        return dataframe
        # X_test = test.drop(columns=dropped_columns, axis=1).values
        # y_test = label_encoder.transform(test['Label'].values)

        # test_data = TensorDataset(torch.tensor(
        #     X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
        # test_loader = DataLoader(test_data, batch_size=32)

        # input_size = X_test.shape[1]
        # num_classes = len(label_encoder.classes_)

        # model = NeuralNetwork(input_size, [64, 32, 16], num_classes)
        # model.load_state_dict(torch.load(pkg_resources.resource_filename("vcab", "best_model.pth")))

        # model.eval()
        # test_logits=[]
        # autism_predictions={}
        # with torch.no_grad():
        #     for inputs, _ in test_loader:
        #         outputs=model(inputs)
        #         test_logits.extend(outputs.numpy())

        # test_probs = np.exp(test_logits) / np.sum(np.exp(test_logits), axis=1, keepdims=True)

        # autism_count=0
        # for i, probs in enumerate(test_probs):
        #     # print(f"Sample {i+1}:")
        #     for class_idx, prob in enumerate(probs):
        #         # print(f"Class {label_encoder.classes_[class_idx]}: {prob * 100:.2f}%")
        #         if label_encoder.classes_[class_idx] == "autism":
        #             time = test.at[i, 'Time']
        #             if prob >= 0.5:
        #                 autism_count += 1
        #                 autism_predictions[time] = "symptoms"
        #             else:
        #                 autism_predictions[time] = "no_symptoms"

        # autism_percentage=autism_count/len(test_probs) * 100

        # return action_predictions, emotion_predictions, autism_predictions, autism_percentage, f"{video_path[:-4]}_emotion.mp4"
