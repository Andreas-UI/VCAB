from vcab import Model, save_video_stream_predictions


def test_predict_stream():
    try:
        video_path = "videos/v_ArmFlapping_01.mp4"
        predictions = Model().predict_stream(video_path=video_path)
        save_video_stream_predictions(
            video_path=video_path, predictions=predictions, output_path="output/v_ArmFlapping_01.mp4")
    except Exception as e:
        assert False, f"Exception occured: {e}"
