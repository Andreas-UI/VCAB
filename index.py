from vcab import Model
from vcab.utils import save_video_stream_predictions_v3

video_path = "videos/v_ArmFlapping_01.mp4"
actions, emotions, autism, autism_percentage, video_output = Model().predict_stream_emotion(video_path=video_path)
save_video_stream_predictions_v3(video_output, actions, autism, video_output)