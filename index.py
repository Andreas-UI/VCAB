from vcab import Model, save_video_stream_predictions

video_path = "videos/v_Spinning_01.mp4"
predictions = Model().predict_stream(video_path=video_path)
save_video_stream_predictions(
    video_path=video_path, predictions=predictions, output_path="output/v_Spinning_01.mp4")
