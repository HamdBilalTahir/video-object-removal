import os
import cv2
import numpy as np
from decord import VideoReader, cpu
from moviepy.editor import VideoFileClip

video_path = "./gradio_demo/normal_videos/0.mp4"
vr = VideoReader(video_path, ctx=cpu(0))
fps = vr.get_avg_fps()
print("FPS:", fps)
duration = len(vr) / fps
print("Duration:", duration)

start_time = 5.0
end_time = 6.5
start_frame = int(start_time * fps)
end_frame = min(int(end_time * fps), len(vr))
print("start_frame:", start_frame, "end_frame:", end_frame)

images = [vr[i].asnumpy() for i in range(start_frame, end_frame)]
height, width = vr[0].shape[0:2]

output_frames_resized = [np.zeros((height, width, 3), dtype=np.uint8) for _ in images]

original_clip = VideoFileClip(video_path)

def fl_make_frame(gf, t):
    frame_idx = int(round(t * fps))
    if start_frame <= frame_idx < end_frame:
        idx = frame_idx - start_frame
        if idx < len(output_frames_resized):
            return output_frames_resized[idx]
    return gf(t)
    
final_clip = original_clip.fl(fl_make_frame, apply_to=["video"])
print("Final clip duration:", final_clip.duration)
final_clip.write_videofile("/tmp/test_stitch.mp4", codec='libx264', audio_codec='aac', verbose=False, logger=None)
print("Done writing!")
