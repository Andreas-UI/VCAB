import numpy as np
import cv2
import ffmpegcv
import warnings


def calculate_font_scale(width, height, font_size, max_font_scale=1):
    base_font_scale = min(width, height) / 1000
    font_scale = (font_size / 10) * base_font_scale
    font_scale = min(font_scale, max_font_scale)
    return font_scale


def save_video_stream_predictions(video_path: str, predictions, output_path="output.mp4") -> None:
    """
    Please use `save_video_stream_prediction_v2 instead`, as this is deprecated.

    Put predictions on top of the video. 
    NOTE:: This function only works from stream prediction output.

    Args:
        video_path <- path to the video
        prediction <- stream predictions
        output_path <- path to the video output, default is output.mp4
    """
    warnings.warn(
        "Please use `save_video_stream_prediction_v2 instead`, as this will be removed in v0.1.1-beta.", FutureWarning)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(
        output_path, fourcc, fps, (frame_width, frame_height))

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 24
    font_scale = calculate_font_scale(frame_width, frame_height, font_size)
    color = (0, 255, 0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # write prediction to the video in the period of time
        for start_sec, end_sec, texts in predictions:
            current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            current_time = current_frame / fps

            if current_time >= start_sec and current_time <= end_sec:
                for i, text in enumerate(texts):
                    max_text_height = max(cv2.getTextSize(
                        text[0], font, font_scale, 1)[0][1] for text in texts)
                    text_x = 10
                    text_y = 10 + max_text_height + i * (max_text_height + 5)
                    cv2.putText(frame, f"{text[0]} {text[1]}", (text_x, text_y),
                                font, font_scale, color, 1, cv2.LINE_AA)

        output_video.write(frame)

    cap.release()
    output_video.release()


def save_video_stream_predictions_v2(video_path: str, predictions, output_path="output.mp4") -> None:
    """
    Put predictions on top of the video. 
    NOTE:: This function only works from stream prediction output.

    Args:
        video_path <- path to the video
        prediction <- stream predictions
        output_path <- path to the video output, default is output.mp4
    """
    cap = ffmpegcv.VideoCapture(video_path)
    fps = cap.fps

    output_video = ffmpegcv.VideoWriter(output_path, None, fps)

    frame_width = int(cap.origin_width)
    frame_height = int(cap.origin_height)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 24
    font_scale = calculate_font_scale(frame_width, frame_height, font_size)
    color = (0, 255, 0)

    for iframe, frame in enumerate(cap):
        frame_copy = np.copy(frame)
        for start_sec, end_sec, texts in predictions:
            current_time = iframe / fps

            if current_time >= start_sec and current_time <= end_sec:
                for i, text in enumerate(texts):
                    max_text_height = max(cv2.getTextSize(
                        text[0], font, font_scale, 1)[0][1] for text in texts)
                    text_x = 10
                    text_y = 10 + max_text_height + i * (max_text_height + 5)
                    cv2.putText(frame_copy, f"{text[0]} {text[1]}", (text_x, text_y),
                                font, font_scale, color, 1, cv2.LINE_AA)

        output_video.write(frame_copy)


def save_video_stream_predictions_v3(video_path: str, action_predictions, autism_predictions, output_path="output.mp4") -> None:
    """
    Put predictions on top of the video. 
    NOTE:: This function only works from stream prediction output.

    Args:
        video_path <- path to the video
        prediction <- stream predictions
        output_path <- path to the video output, default is output.mp4
    """
    action_map = {
        "armflapping": "Arm Flapping",
        "spinning": "Spinning",
        "headbanging": "Head Banging"
    }

    combined_predictions = []
    start_sec = None
    for time, label in autism_predictions.items():
        for start_sec_act, end_sec_act, actions in action_predictions:
            if start_sec_act <= time <= end_sec_act:
                if label == "symptoms":
                    combined_predictions.append((start_sec_act if start_sec is None else start_sec, time, label, [
                                                action for action in actions if action[0] != 'normal']))
                elif label == "no_symptoms":
                    combined_predictions.append((start_sec_act if start_sec is None else start_sec, time, label, [
                                                action for action in actions if action[0] == 'normal']))

        if start_sec is None:
            start_sec = time
        else:
            start_sec = time

    cap = ffmpegcv.VideoCapture(video_path)
    fps = cap.fps

    output_video = ffmpegcv.VideoWriter(output_path, None, fps)

    frame_width = int(cap.origin_width)
    frame_height = int(cap.origin_height)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 24
    font_scale = calculate_font_scale(frame_width, frame_height, font_size)
    color = (250, 122, 2)

    for iframe, frame in enumerate(cap):
        frame_copy = np.copy(frame)
        for start_sec, end_sec, label, texts in combined_predictions:
            current_time = iframe / fps

            if current_time >= start_sec and current_time <= end_sec:
                text = "No Symptoms Shown" if label == "no_symptoms" else action_map[texts[0][0]]
                text_size, _ = cv2.getTextSize(
                    text, font, font_scale, 1)
                
                text_x = 10
                text_y = 10 + text_size[1]

                bg_padding = 10
                x = text_x - bg_padding
                y = text_y - text_size[1] - bg_padding
                w = text_size[0] + 2 * bg_padding
                h = text_size[1] + 2 * bg_padding
                sub_img = frame_copy[y:y+h, x:x+w]
                white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
                res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)
                frame_copy[y:y+h, x:x+w] = res

                cv2.putText(frame_copy, text, (text_x, text_y),
                            font, font_scale, color, 1, cv2.LINE_AA)

                # for i, text in enumerate([label] + texts):
                #     max_text_height = max(cv2.getTextSize(text[0], font, font_scale, 1)[0][1] for text in texts)
                #     text_x = 10
                #     text_y = 10 + max_text_height + i * (max_text_height + 5)

                #     cv2.putText(frame_copy, label if i == 0 else f"{text[0]} {text[1]}", (text_x, text_y),
                #                 font, font_scale, color, 1, cv2.LINE_AA)

        output_video.write(frame_copy)
