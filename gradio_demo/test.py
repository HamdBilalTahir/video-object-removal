import os
import shutil
import gradio as gr
import cv2
import numpy as np
from PIL import Image

STORAGE_DIR = "./storage"
os.makedirs(STORAGE_DIR, exist_ok=True)

os.makedirs("./SAM2-Video-Predictor/checkpoints/", exist_ok=True)
os.makedirs("./model/", exist_ok=True)

from huggingface_hub import snapshot_download

def download_sam2():
    if not os.path.exists("./SAM2-Video-Predictor/checkpoints/sam2_hiera_large.pt"):
        snapshot_download(repo_id="facebook/sam2-hiera-large", local_dir="./SAM2-Video-Predictor/checkpoints/")
        print("Download sam2 completed")
    else:
        print("SAM2 already downloaded, skipping.")

download_sam2()

import torch
import random
import time
from decord import VideoReader, cpu
from moviepy.editor import ImageSequenceClip
from iopaint.model import LaMa
from iopaint.schema import InpaintRequest, HDStrategy

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

COLOR_PALETTE = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (255, 128, 0), (128, 0, 255), (0, 128, 255), (128, 255, 0)
]

random_seed = 42
video_length = 201
W = 1024
H = W
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_lama_and_predictors():
    lama = LaMa(device=torch.device(device))

    sam2_checkpoint = "./SAM2-Video-Predictor/checkpoints/sam2_hiera_large.pt"
    config = "sam2_hiera_l.yaml"

    video_predictor = build_sam2_video_predictor(config, sam2_checkpoint, device=device)
    model = build_sam2(config, sam2_checkpoint, device=device)
    model.image_size = 1024
    image_predictor = SAM2ImagePredictor(sam_model=model)

    return lama, image_predictor, video_predictor

def get_video_info(video_path, current_time, video_state):
    video_state["input_points"] = []
    video_state["scaled_points"] = []
    video_state["input_labels"] = []
    video_state["frame_idx"] = 0
    vr = VideoReader(video_path, ctx=cpu(0))
    fps = vr.get_avg_fps()
    frame_idx = min(int(current_time * fps), len(vr) - 1)
    first_frame = vr[frame_idx].asnumpy()
    del vr

    if first_frame.shape[0] > first_frame.shape[1]:
        W_ = W
        H_ = int(W_ * first_frame.shape[0] / first_frame.shape[1])
    else:
        H_ = H
        W_ = int(H_ * first_frame.shape[1] / first_frame.shape[0])

    first_frame = cv2.resize(first_frame, (W_, H_))
    video_state["origin_images"] = np.expand_dims(first_frame, axis=0)
    video_state["inference_state"] = None
    video_state["video_path"] = video_path
    video_state["masks"] = None
    video_state["painted_images"] = None
    image = Image.fromarray(first_frame)
    return image

def segment_frame(evt: gr.SelectData, label, video_state):
    if video_state["origin_images"] is None:
        gr.Warning("Please click \"Extract First Frame\" to extract the first frame first, then click the annotation")
        return None
    x, y = evt.index
    new_point = [x, y]
    label_value = 1 if label == "Positive" else 0

    video_state["input_points"].append(new_point)
    video_state["input_labels"].append(label_value)
    height, width = video_state["origin_images"][0].shape[0:2]
    scaled_points = []
    for pt in video_state["input_points"]:
        sx = pt[0] / width
        sy = pt[1] / height
        scaled_points.append([sx, sy])

    video_state["scaled_points"] = scaled_points

    image_predictor.set_image(video_state["origin_images"][0])
    mask, _, _ = image_predictor.predict(
        point_coords=video_state["scaled_points"],
        point_labels=video_state["input_labels"],
        multimask_output=False,
        normalize_coords=False,
    )

    mask = np.squeeze(mask)
    mask = mask.astype(np.float32)
    mask = cv2.resize(mask, (width, height))
    mask = mask[:,:,None]

    color = np.array(COLOR_PALETTE[int(time.time()) % len(COLOR_PALETTE)], dtype=np.float32) / 255.0
    color = color[None, None, :]
    org_image = video_state["origin_images"][0].astype(np.float32) / 255.0
    painted_image = (1 - mask * 0.5) * org_image + mask * 0.5 * color
    painted_image = np.uint8(np.clip(painted_image * 255, 0, 255))
    video_state["painted_images"] = np.expand_dims(painted_image, axis=0)
    video_state["masks"] = np.expand_dims(mask[:,:,0], axis=0)

    for i in range(len(video_state["input_points"])):
        point = video_state["input_points"][i]
        if video_state["input_labels"][i] == 0:
            cv2.circle(painted_image, point, radius=3, color=(0, 0, 255), thickness=-1)  # 红色点，半径为3
        else:
            cv2.circle(painted_image, point, radius=3, color=(255, 0, 0), thickness=-1)

    return Image.fromarray(painted_image)

def clear_clicks(video_state):
    video_state["input_points"] = []
    video_state["input_labels"] = []
    video_state["scaled_points"] = []
    video_state["inference_state"] = None
    video_state["masks"] = None
    video_state["painted_images"] = None
    return Image.fromarray(video_state["origin_images"][0]) if video_state["origin_images"] is not None else None


from moviepy.editor import VideoFileClip

def inference_and_return_video(dilation_iterations, use_gpu, video_state=None, progress=gr.Progress()):
    if video_state["origin_images"] is None or video_state["masks"] is None:
        return None, gr.update(interactive=False)
    if len(video_state["origin_images"]) == 1 and video_state.get("start_frame") is None:
        gr.Warning("Please run 'Tracking' first before clicking 'Remove'!")
        return None, gr.update(interactive=False)
    images = video_state["origin_images"]
    masks = video_state["masks"]
    total = len(images)
    
    video_path = video_state["video_path"]
    start_frame = video_state["start_frame"]
    end_frame = video_state["end_frame"]
    original_height = video_state["original_height"]
    original_width = video_state["original_width"]
    fps = video_state["fps"]

    output_frames = []
    
    if use_gpu:
        progress(0.1, desc="Loading MiniMax-Remover pipeline (GPU)...")
        try:
            from diffusers.models import AutoencoderKLWan
            from transformer_minimax_remover import Transformer3DModel
            from diffusers.schedulers import UniPCMultistepScheduler
            from pipeline_minimax_remover import Minimax_Remover_Pipeline
            
            vae = AutoencoderKLWan.from_pretrained("../vae", torch_dtype=torch.float16)
            transformer = Transformer3DModel.from_pretrained("../transformer", torch_dtype=torch.float16)
            scheduler = UniPCMultistepScheduler.from_pretrained("../scheduler")
            
            pipe = Minimax_Remover_Pipeline(
                vae=vae, transformer=transformer, scheduler=scheduler, torch_dtype=torch.float16
            ).to("cuda:0")
            
            progress(0.3, desc="Running MiniMax-Remover (GPU)...")
            
            # images expects values in [-1, 1], and shape (T, H, W, C)
            images_tensor = (torch.from_numpy(np.array(images)).float() / 127.5) - 1.0
            
            # mask needs shape (T, H, W, 1) and in [0, 1]
            mask_list = []
            for m in masks:
                if m.ndim == 3:
                    m = m[:, :, 0]
                # apply dilation
                kernel = np.ones((dilation_iterations * 2 + 1, dilation_iterations * 2 + 1), np.uint8)
                m_uint8 = (m > 0.5).astype(np.uint8)
                m_dilated = cv2.dilate(m_uint8, kernel, iterations=1)
                mask_list.append(m_dilated)
                
            masks_tensor = torch.from_numpy(np.array(mask_list)).float()
            if masks_tensor.ndim == 3:
                masks_tensor = masks_tensor.unsqueeze(-1)
                
            result_frames = pipe(
                images=images_tensor, 
                masks=masks_tensor, 
                num_frames=len(images), 
                height=original_height, 
                width=original_width, 
                num_inference_steps=12, 
                generator=torch.Generator(device="cuda:0").manual_seed(42), 
                iterations=0 # already dilated above
            ).frames[0]
            
            output_frames = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) if hasattr(cv2, "COLOR_RGB2BGR") else f for f in result_frames]
            
        except Exception as e:
            gr.Warning(f"Failed to run GPU pipeline: {str(e)}. Make sure weights are downloaded to the root directory.")
            return None
    else:
        lama_config = InpaintRequest(
            hd_strategy=HDStrategy.ORIGINAL,
            hd_strategy_crop_trigger_size=800,
            hd_strategy_crop_margin=32,
            hd_strategy_resize_limit=1280,
        )

        for i, (img, msk) in enumerate(zip(images, masks)):
            progress(i / total, desc=f"Inpainting frame {i+1}/{total} (CPU)...")
            # img: uint8 RGB (H,W,3), msk: float32 (H,W) or (H,W,C)
            img_uint8 = img.astype(np.uint8)
            if msk.ndim == 3:
                msk = msk[:, :, 0]
            
            # Ensure mask is float32
            msk = msk.astype(np.float32)

            # Dilate mask
            kernel = np.ones((dilation_iterations * 2 + 1, dilation_iterations * 2 + 1), np.uint8)
            msk_uint8 = (msk > 0.5).astype(np.uint8) * 255
            msk_uint8 = msk_uint8.astype(np.uint8)
            msk_uint8 = cv2.dilate(msk_uint8, kernel, iterations=1)
            result = lama_model(img_uint8, msk_uint8, lama_config)
            # LaMa returns float64 BGR, so convert to uint8 RGB for ImageSequenceClip
            result = np.clip(result, 0, 255).astype(np.uint8)
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            output_frames.append(result)

    progress(0.85, desc="Stitching full video...")
    
    video_path = video_state["video_path"]
    start_frame = video_state["start_frame"]
    end_frame = video_state["end_frame"]
    original_height = video_state["original_height"]
    original_width = video_state["original_width"]
    fps = video_state["fps"]

    # Load the original video clip
    original_clip = VideoFileClip(video_path)
    
    # Resize the inpainted frames back to original resolution
    output_frames_resized = [cv2.resize(f, (original_width, original_height)) for f in output_frames]
    
    # We create a function to replace frames in the moviepy clip
    def fl_make_frame(gf, t):
        frame_idx = int(round(t * fps))
        if start_frame <= frame_idx < end_frame:
            idx = frame_idx - start_frame
            if idx < len(output_frames_resized):
                return output_frames_resized[idx]
        return gf(t)
        
    final_clip = original_clip.fl(fl_make_frame, apply_to=["video"])

    progress(0.95, desc="Encoding video...")
    video_file_tmp = f"{STORAGE_DIR}/{time.time()}-{random.random()}-tmp.mp4"
    video_file = f"{STORAGE_DIR}/{time.time()}-{random.random()}-removed_output.mp4"
    
    # Use -crf 18 for visually lossless video, audio=False because we will mux original audio
    final_clip.write_videofile(video_file_tmp, codec='libx264', audio=False, ffmpeg_params=["-crf", "18"], verbose=False, logger=None)
    
    original_clip.close()
    final_clip.close()
    
    progress(0.98, desc="Muxing original audio...")
    # Directly copy audio stream from original video to preserve exact audio quality
    os.system(f"ffmpeg -y -i {video_file_tmp} -i {video_path} -c:v copy -c:a copy -map 0:v:0 -map 1:a:0? {video_file}")
    
    if os.path.exists(video_file_tmp):
        os.remove(video_file_tmp)
        
    progress(1.0, desc="Done!")
    return video_file


def get_video_duration(video_path):
    if video_path is None:
        return gr.update(), gr.update(), "<p style='color:#aaa'>No video loaded.</p>"
    vr = VideoReader(video_path, ctx=cpu(0))
    fps = vr.get_avg_fps()
    duration = len(vr) / fps
    del vr
    ruler_html = _build_ruler_html(duration)
    return (
        gr.update(maximum=duration, value=0),
        gr.update(maximum=duration, value=duration),
        ruler_html,
    )


def _build_ruler_html(duration):
    total = max(duration, 1)
    ticks_html = ""
    step = 1 if total <= 30 else (5 if total <= 120 else 10)
    t = 0
    while t <= total:
        pct = t / total * 100
        ticks_html += (
            f"<div style='position:absolute;left:{pct:.2f}%;transform:translateX(-50%);text-align:center;'>"
            f"<div style='width:1px;height:8px;background:#888;margin:0 auto'></div>"
            f"<span style='font-size:10px;color:#aaa'>{t:.0f}s</span></div>"
        )
        t += step
    return (
        f"<div id='timeline-ruler' style='position:relative;width:100%;height:32px;background:#222;"
        f"border-radius:4px;overflow:visible;margin:6px 0'>{ticks_html}</div>"
    )


def track_video(start_time, end_time, video_state, progress=gr.Progress()):
    if video_state["origin_images"] is None or video_state["masks"] is None:
        gr.Warning("Please complete target segmentation on the first frame first, then click Tracking")
        return None

    input_points = video_state["input_points"]
    input_labels = video_state["input_labels"]
    frame_idx = video_state["frame_idx"]
    obj_id = video_state["obj_id"]
    scaled_points = video_state["scaled_points"]

    progress(0, desc="Loading video frames...")
    vr = VideoReader(video_state["video_path"], ctx=cpu(0))
    fps = vr.get_avg_fps()
    start_frame = int(start_time * fps)
    end_frame = min(int(end_time * fps), len(vr))
    height, width = vr[0].shape[0:2]
    images = [vr[i].asnumpy() for i in range(start_frame, end_frame)]
    del vr

    if images[0].shape[0] > images[0].shape[1]:
        W_ = W
        H_ = int(W_ * images[0].shape[0] / images[0].shape[1])
    else:
        H_ = H
        W_ = int(H_ * images[0].shape[1] / images[0].shape[0])

    progress(0.1, desc="Resizing frames...")
    images = [cv2.resize(img, (W_, H_)) for img in images]
    video_state["origin_images"] = images
    video_state["start_frame"] = start_frame
    video_state["end_frame"] = end_frame
    video_state["fps"] = fps
    video_state["original_height"] = height
    video_state["original_width"] = width
    images = np.array(images)

    progress(0.2, desc="Initialising SAM2 tracker...")
    inference_state = video_predictor.init_state(images=images/255, device=device)
    video_state["inference_state"] = inference_state

    if len(torch.from_numpy(video_state["masks"][0]).shape) == 3:
        mask = torch.from_numpy(video_state["masks"][0])[:,:,0]
    else:
        mask = torch.from_numpy(video_state["masks"][0])

    video_predictor.add_new_mask(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=obj_id,
        mask=mask
    )

    output_frames = []
    mask_frames = []
    total_frames = len(images)
    color = np.array(COLOR_PALETTE[int(time.time()) % len(COLOR_PALETTE)], dtype=np.float32) / 255.0
    color = color[None, None, :]
    progress(0.3, desc=f"Tracking {total_frames} frames...")
    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
        frame = images[out_frame_idx].astype(np.float32) / 255.0
        mask = np.zeros((H, W, 3), dtype=np.float32)
        for i, logit in enumerate(out_mask_logits):
            out_mask = logit.cpu().squeeze().detach().numpy()
            out_mask = (out_mask[:,:,None] > 0).astype(np.float32)
            mask += out_mask
        mask = np.clip(mask, 0, 1)
        mask = cv2.resize(mask, (W_, H_))
        mask_frames.append(mask)
        painted = (1 - mask * 0.5) * frame + mask * 0.5 * color
        painted = np.uint8(np.clip(painted * 255, 0, 255))
        output_frames.append(painted)
        progress(0.3 + 0.6 * (out_frame_idx + 1) / total_frames,
                 desc=f"Tracking frame {out_frame_idx + 1}/{total_frames}...")
    video_state["masks"] = mask_frames
    progress(0.95, desc="Encoding video...")
    video_file = f"{STORAGE_DIR}/{time.time()}-{random.random()}-tracked_output.mp4"
    clip = ImageSequenceClip(output_frames, fps=fps)
    clip.write_videofile(video_file, codec='libx264', audio=False, ffmpeg_params=["-crf", "18"], verbose=False, logger=None)
    progress(1.0, desc="Done!")
    return video_file

text = """
<div style='text-align:center; font-size:32px; font-family: Arial, Helvetica, sans-serif;'>
  Minimax-Remover: Taming Bad Noise Helps Video Object Removal
</div>
<div style="display: flex; justify-content: center; align-items: center; gap: 10px; flex-wrap: nowrap;">
  <a href="https://huggingface.co/zibojia/minimax-remover"><img alt="Huggingface Model" src="https://img.shields.io/badge/%F0%9F%A4%97%20Huggingface-Model-brightgreen"></a>
  <a href="https://github.com/zibojia/MiniMax-Remover"><img alt="Github" src="https://img.shields.io/badge/MiniMaxRemover-github-black"></a>
  <a href="https://huggingface.co/spaces/zibojia/MiniMaxRemover"><img alt="Huggingface Space" src="https://img.shields.io/badge/%F0%9F%A4%97%20Huggingface-Space-1e90ff"></a>
  <a href="https://arxiv.org/abs/2505.24873"><img alt="arXiv" src="https://img.shields.io/badge/MiniMaxRemover-arXiv-b31b1b"></a>
  <a href="https://www.youtube.com/watch?v=KaU5yNl6CTc"><img alt="YouTube" src="https://img.shields.io/badge/Youtube-video-ff0000"></a>
  <a href="https://minimax-remover.github.io"><img alt="Demo Page" src="https://img.shields.io/badge/Website-Demo%20Page-yellow"></a>
</div>
<div style='text-align:center; font-size:20px; margin-top: 10px; font-family: Arial, Helvetica, sans-serif;'>
  Bojia Zi<sup>*</sup>, Weixuan Peng<sup>*</sup>, Xianbiao Qi<sup>†</sup>, Jianan Wang, Shihao Zhao, Rong Xiao, Kam-Fai Wong
</div>
<div style='text-align:center; font-size:14px; color: #888; margin-top: 5px; font-family: Arial, Helvetica, sans-serif;'>
  <sup>*</sup> Equal contribution &nbsp; &nbsp; <sup>†</sup> Corresponding author
</div>
"""

lama_model, image_predictor, video_predictor = get_lama_and_predictors()

with gr.Blocks() as demo:
    video_state = gr.State({
        "origin_images": None,
        "inference_state": None,
        "masks": None,  # Store user-generated masks
        "painted_images": None,
        "video_path": None,
        "input_points": [],
        "scaled_points": [],
        "input_labels": [],
        "frame_idx": 0,
        "obj_id": 1
    })
    gr.Markdown(f"<div style='text-align:center;'>{text}</div>")

    with gr.Column():
        video_input = gr.Video(
            label="Upload Video",
            elem_id="my-video1",
            height=300
        )
        
        def on_upload_copy(vp):
            if not vp:
                return gr.update()
            
            # Using realpath to avoid macOS /private symlink mismatches
            abs_storage = os.path.realpath(STORAGE_DIR)
            vp_abs = os.path.realpath(vp)
            
            # Break infinite loop if already in storage or if it's an example
            if vp_abs.startswith(abs_storage) or "cartoon" in vp or "normal_videos" in vp:
                # We return gr.update() to skip updating the component, completely breaking the infinite loop
                return gr.update()
                
            name = os.path.basename(vp)
            new_path = os.path.join(STORAGE_DIR, f"{int(time.time())}_{name}")
            shutil.copy2(vp, new_path)
            return new_path
        
        video_input.change(fn=on_upload_copy, inputs=[video_input], outputs=[video_input])
        gr.Markdown(
            "<div style='text-align:center;color:#aaa;font-size:12px'>"
            "📌 <b>How to use:</b> Upload a video → pause at the frame with the object → click <i>Extract Current Frame</i> → click the object in the image → adjust Start/End time → click <i>Tracking</i> → click <i>Remove</i>"
            "</div>",
            elem_id="my-btn"
        )
        current_time_box = gr.Number(value=0, visible=False)
        get_info_btn = gr.Button("Extract Current Frame", elem_id="my-btn")

        gr.Examples(
            examples=[
                ["./cartoon/0.mp4"],
                ["./cartoon/1.mp4"],
                ["./cartoon/2.mp4"],
                ["./cartoon/3.mp4"],
                ["./cartoon/4.mp4"],
                ["./normal_videos/0.mp4"],
                ["./normal_videos/1.mp4"],
                ["./normal_videos/3.mp4"],
                ["./normal_videos/4.mp4"],
                ["./normal_videos/5.mp4"],
            ],
            inputs=[video_input],
            label="Choose a video to remove.",
            elem_id="my-btn2"
        )

        image_output = gr.Image(label="First Frame Segmentation", interactive=True, elem_id="my-video", height=400)
        demo.css = """
        #my-btn {
           width: 60% !important;
           margin: 0 auto;
        }

        #my-video1 {
           width: 60% !important;
           max-height: 280px !important;
           margin: 0 auto;
        }
        #my-video1 video {
           max-height: 240px !important;
           object-fit: contain !important;
        }
        #my-video {
           width: 60% !important;
           max-height: 320px !important;
           margin: 0 auto;
        }
        #my-video video {
           max-height: 260px !important;
           object-fit: contain !important;
        }
        #my-md {
           margin: 0 auto;
        }
        #my-btn2 {
            width: 60% !important;
            margin: 0 auto;
        }
        #my-btn2 button {
            width: 120px !important;
            max-width: 120px !important;
            min-width: 120px !important;
            height: 70px !important;
            max-height: 70px !important;
            min-height: 70px !important;
            margin: 8px !important;
            border-radius: 8px !important;
            overflow: hidden !important;
            white-space: normal !important;
        }
        /* Thicker, more interactive video scrubber */
        video::-webkit-media-controls-timeline {
            height: 10px !important;
            padding: 0 !important;
        }
        .video-container input[type=range],
        video ~ * input[type=range] {
            height: 10px !important;
            cursor: pointer !important;
        }
        input[type=range]::-webkit-slider-runnable-track {
            height: 8px !important;
            border-radius: 4px !important;
        }
        input[type=range]::-webkit-slider-thumb {
            width: 10px !important;
            height: 28px !important;
            margin-top: -10px !important;
            border-radius: 3px !important;
            background: white !important;
            border: 1px solid #ccc !important;
            box-shadow: 0 1px 4px rgba(0,0,0,0.4) !important;
            cursor: pointer !important;
        }
        input[type=range]::-moz-range-thumb {
            width: 10px !important;
            height: 28px !important;
            border-radius: 3px !important;
            background: white !important;
            border: 1px solid #ccc !important;
            box-shadow: 0 1px 4px rgba(0,0,0,0.4) !important;
            cursor: pointer !important;
        }
        """
        with gr.Row(elem_id="my-btn"):
            point_prompt = gr.Radio(
                ["Positive", "Negative"],
                label="Click Type",
                value="Positive",
                info="Positive = click ON the object you want to remove. Negative = click on areas you do NOT want selected (useful to exclude accidentally selected regions)."
            )
            clear_btn = gr.Button("Clear All Clicks")

        timeline_ruler = gr.HTML("<div style='color:#aaa;font-size:12px;padding:6px 0'>Upload a video to see the timeline ruler.</div>", elem_id="my-btn")
        with gr.Row(elem_id="my-btn"):
            start_time_slider = gr.Slider(
                minimum=0, maximum=60, value=0, step=0.5,
                label="Start Time (s)",
                info="The second in the video where tracking begins. The object mask from the extracted frame is propagated forward from this point. Earlier start = more frames to process = slower & more RAM."
            )
            end_time_slider = gr.Slider(
                minimum=0, maximum=60, value=10, step=0.5,
                label="End Time (s)",
                info="The second in the video where tracking stops. Shorter ranges (fewer seconds between start and end) are faster and use less RAM. Each extra second at 30fps adds ~30 frames to track."
            )
        with gr.Row(elem_id="my-btn"):
            track_btn = gr.Button("Tracking")
        video_output = gr.Video(label="Tracking Result", elem_id="my-video", height=300)

        with gr.Column(elem_id="my-btn"):
            dilation_slider = gr.Slider(
                minimum=1, maximum=20, value=6, step=1,
                label="Mask Dilation",
                info="Expands the removal mask outward by this many pixels before inpainting. Higher values cover more area around the object's edges (good for fuzzy or moving edges), but may remove too much background. Recommended: 4–8. No impact on speed or RAM."
            )

        with gr.Row(elem_id="my-btn"):
            use_gpu_checkbox = gr.Checkbox(label="Use GPU (MiniMax-Remover Pipeline)", value=False, info="Uncheck to use the CPU-friendly LaMa fallback.")
        remove_btn = gr.Button("Remove", elem_id="my-btn")
        remove_video = gr.Video(label="Remove Results", elem_id="my-video", height=300)
        
        with gr.Row(elem_id="my-btn"):
            download_btn = gr.DownloadButton("Download Output Video", interactive=False)
        
        # Load state from localStorage on page load
        js_load = """
        async () => {
            try {
                const up = JSON.parse(localStorage.getItem('minimax_up'));
                const tr = JSON.parse(localStorage.getItem('minimax_tr'));
                const rm = JSON.parse(localStorage.getItem('minimax_rm'));
                return [up || null, tr || null, rm || null];
            } catch(e) {
                return [null, null, null];
            }
        }
        """
        demo.load(fn=None, inputs=[], outputs=[video_input, video_output, remove_video], js=js_load)

        # Save state to localStorage on change
        js_save_up = "(val) => { if (val) localStorage.setItem('minimax_up', JSON.stringify(val)); else localStorage.removeItem('minimax_up'); }"
        js_save_tr = "(val) => { if (val) localStorage.setItem('minimax_tr', JSON.stringify(val)); else localStorage.removeItem('minimax_tr'); }"
        js_save_rm = "(val) => { if (val) localStorage.setItem('minimax_rm', JSON.stringify(val)); else localStorage.removeItem('minimax_rm'); }"
        
        video_input.change(fn=None, inputs=[video_input], outputs=[], js=js_save_up)
        video_output.change(fn=None, inputs=[video_output], outputs=[], js=js_save_tr)
        remove_video.change(fn=None, inputs=[remove_video], outputs=[], js=js_save_rm)
        def enable_download(video):
            if video:
                return gr.update(value=video, interactive=True)
            return gr.update(interactive=False)

        remove_btn.click(
            inference_and_return_video,
            inputs=[dilation_slider, use_gpu_checkbox, video_state],
            outputs=remove_video
        ).then(
            fn=enable_download,
            inputs=[remove_video],
            outputs=[download_btn]
        )
        get_info_btn.click(
            fn=get_video_info,
            inputs=[video_input, current_time_box, video_state],
            outputs=image_output,
            js="(vp, t, vs) => [vp, (document.querySelector('#my-video1 video')?.currentTime ?? 0), vs]"
        )
        video_input.change(
            get_video_duration,
            inputs=[video_input],
            outputs=[start_time_slider, end_time_slider, timeline_ruler]
        )
        image_output.select(fn=segment_frame, inputs=[point_prompt, video_state], outputs=image_output)
        clear_btn.click(clear_clicks, inputs=video_state, outputs=image_output)
        track_btn.click(track_video, inputs=[start_time_slider, end_time_slider, video_state], outputs=video_output)

demo.launch(server_name="0.0.0.0", server_port=8000, allowed_paths=["/tmp", STORAGE_DIR], share=True)
