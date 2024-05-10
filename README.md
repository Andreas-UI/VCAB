# How to use
```python
from vcab import Model, save_video_stream_predictions_v2

video_path = "example.mp4"

# Get video stream predictions
actions, emotions, autism, autism_percentage, video_output = Model().predict_stream_emotion(video_path=video_path)

# Mask video with prediction
save_video_stream_predictions_v2(
    video_path=video_path,
    action_predictions=actions,
    autism_predictions=autism, 
    output_path="example_output.mp4")
```

# Requirements
Built under python 3.10.11 and tested on python 3.9 >=, 3.11.5 <=

# Important
Please install ffmpeg on your machine to use this package. [link](https://www.hostinger.my/tutorials/how-to-install-ffmpeg)
