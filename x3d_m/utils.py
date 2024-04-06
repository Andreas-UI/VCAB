import cv2


def calculate_font_scale(width, height, font_size, max_font_scale=1):
    base_font_scale = min(width, height) / 1000
    font_scale = (font_size / 10) * base_font_scale
    font_scale = min(font_scale, max_font_scale)
    return font_scale


def save_video_stream_predictions(video_path: str, predictions, output_path="output.mp4") -> None:
    """
    Put predictions on top of the video. 
    NOTE:: This function only works from stream prediction output.

    Args:
        video_path <- path to the video
        prediction <- stream predictions
        output_path <- path to the video output, default is output.mp4
    """
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
                    text_size = cv2.getTextSize(
                        text[0], font, font_scale, 1)[0]
                    text_x = 10
                    text_y = 10 + i * (text_size[1] + 5)
                    cv2.putText(frame, f"{text[0]} {text[1]}", (text_x, text_y),
                                font, font_scale, color, 1, cv2.LINE_AA)

        output_video.write(frame)

    cap.release()
    output_video.release()
