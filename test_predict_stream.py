from x3d_m import Model, save_video_stream_predictions


def test_predict_stream():
    try:
        video_path = r"videos/v_ArmFlapping_01.mp4"
        predictions = Model().predict_stream(video_path=video_path)
        save_video_stream_predictions(
            video_path=video_path, predictions=predictions, output_path=r"output/v_ArmFlapping_01.mp4")
    except Exception as e:
        assert False, f"Exception occured: {e}"