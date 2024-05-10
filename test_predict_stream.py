from vcab import Model, save_video_stream_predictions_v3

# def test_predict_stream():
#     try:
#         video_path = "videos/v_ArmFlapping_01.mp4"
#         predictions = Model().predict_stream(video_path=video_path)
#         save_video_stream_predictions_v2(
#             video_path=video_path, predictions=predictions, output_path="output/v_ArmFlapping_01.mp4")
#     except Exception as e:
#         assert False, f"Exception occured: {e}"

def test_multimodal():
    try:
        video_path = "videos/v_ArmFlapping_01.mp4"
        actions, emotions, autism, autism_percentage, video_output = Model().predict_stream_emotion(video_path=video_path)
        save_video_stream_predictions_v3(video_output, actions, autism, video_output)
        print(autism_percentage)
    except Exception as e:
        assert False, f"Exception occured: {e}"
    