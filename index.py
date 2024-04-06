from vcab import Model, save_video_stream_predictions

video_path = r"v_ArmFlapping_01.mp4"
predictions = Model().predict_stream(video_path=video_path)
save_video_stream_predictions(
    video_path=video_path, predictions=predictions, output_path="v_ArmFlapping_output.mp4")
