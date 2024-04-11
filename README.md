# How to use
```python
from vcab import Model, save_video_stream_predictions_v2

video_path = "example.mp4"

# Get video stream predictions
predictions = Model().predict_stream(video_path=video_path)

# Mask video with prediction
save_video_stream_predictions_v2(
    video_path=video_path,
    predictions=predictions, 
    output_path="example_output.mp4")
```

# Requirements
Built under python 3.10.11