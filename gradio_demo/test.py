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

    try:
        from iopaint.model import LaMaONNX
        lama_onnx = LaMaONNX(device=torch.device("cpu"))
    except Exception:
        lama_onnx = None

    sam2_checkpoint = "./SAM2-Video-Predictor/checkpoints/sam2_hiera_large.pt"
    config = "sam2_hiera_l.yaml"

    video_predictor = build_sam2_video_predictor(config, sam2_checkpoint, device=device)
    model = build_sam2(config, sam2_checkpoint, device=device)
    model.image_size = 1024
    image_predictor = SAM2ImagePredictor(sam_model=model)

    return lama, lama_onnx, image_predictor, video_predictor

def get_video_info(video_path, current_time, video_state):
    # Prefer the durable storage copy set by on_upload_copy over the Gradio temp path,
    # which may be cleaned up before or during long operations.
    video_path = video_state.get("video_path") or video_path
    if not video_path or not os.path.isfile(video_path):
        return None, None
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
    return image, image

def segment_frame(evt: gr.SelectData, label, video_state, current_mask_color):
    if video_state["origin_images"] is None:
        gr.Warning("Please click \"Extract First Frame\" to extract the first frame first, then click the annotation")
        return None, current_mask_color
    x, y = evt.index

    if label == "Pick Color":
        # Get color at x, y. origin_images[0] is RGB, y is row, x is col
        pixel = video_state["origin_images"][0][y, x]
        hex_color = f"#{pixel[0]:02x}{pixel[1]:02x}{pixel[2]:02x}"
        current_img = video_state["painted_images"][0] if video_state["painted_images"] is not None else video_state["origin_images"][0]
        return Image.fromarray(current_img), hex_color

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

    return Image.fromarray(painted_image), current_mask_color

def clear_clicks(video_state):
    video_state["input_points"] = []
    video_state["input_labels"] = []
    video_state["scaled_points"] = []
    video_state["inference_state"] = None
    video_state["masks"] = None
    video_state["painted_images"] = None
    return Image.fromarray(video_state["origin_images"][0]) if video_state["origin_images"] is not None else None


from moviepy.editor import VideoFileClip

def inference_and_return_video(dilation_iterations, inpaint_mode, telea_radius, mask_color="#000000", bg_video_path=None, bg_image_path=None, video_state=None, progress=gr.Progress()):
    if video_state["origin_images"] is None or video_state["masks"] is None:
        return None
    if len(video_state["origin_images"]) == 1 and video_state.get("start_frame") is None:
        gr.Warning("Please run 'Tracking' first before processing!")
        return None
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

    if inpaint_mode == "Solid Color":
        # Parse hex color to RGB tuple
        hex_color = mask_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        color_arr = np.array(rgb, dtype=np.uint8)

        for i, (img, msk) in enumerate(zip(images, masks)):
            progress(i / total, desc=f"Processing frame {i+1}/{total} (Solid Color)...")
            img_uint8 = img.astype(np.uint8)
            if msk.ndim == 3:
                msk = msk[:, :, 0]
            msk = msk.astype(np.float32)

            kernel = np.ones((dilation_iterations * 2 + 1, dilation_iterations * 2 + 1), np.uint8)
            msk_uint8 = (msk > 0.5).astype(np.uint8)
            msk_dilated = cv2.dilate(msk_uint8, kernel, iterations=1)

            # Create solid color background for mask
            mask_3ch = msk_dilated[:, :, None]
            result = img_uint8 * (1 - mask_3ch) + color_arr * mask_3ch
            result = result.astype(np.uint8)

            output_frames.append(result)

    elif inpaint_mode == "Video Background":
        if not bg_video_path or not os.path.isfile(bg_video_path):
            gr.Warning("Please upload a background video.")
            return None

        bg_vr = VideoReader(bg_video_path, ctx=cpu(0))
        bg_len = len(bg_vr)
        for i, (img, msk) in enumerate(zip(images, masks)):
            progress(i / total, desc=f"Processing frame {i+1}/{total} (Video Background)...")
            img_uint8 = img.astype(np.uint8)
            if msk.ndim == 3:
                msk = msk[:, :, 0]
            msk = msk.astype(np.float32)

            kernel = np.ones((dilation_iterations * 2 + 1, dilation_iterations * 2 + 1), np.uint8)
            msk_uint8 = (msk > 0.5).astype(np.uint8)
            msk_dilated = cv2.dilate(msk_uint8, kernel, iterations=1)
            mask_3ch = msk_dilated[:, :, None]

            # Get corresponding frame from background video
            bg_idx = min(i, bg_len - 1)
            bg_frame = bg_vr[bg_idx].asnumpy()
            bg_frame_resized = cv2.resize(bg_frame, (img.shape[1], img.shape[0]))

            result = img_uint8 * (1 - mask_3ch) + bg_frame_resized * mask_3ch
            result = result.astype(np.uint8)

            output_frames.append(result)
        del bg_vr

    elif inpaint_mode == "Image Background":
        if not bg_image_path or not os.path.isfile(bg_image_path):
            gr.Warning("Please upload a background image.")
            return None

        bg_img = cv2.imread(bg_image_path)
        if bg_img is None:
            gr.Warning("Failed to load background image.")
            return None
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)

        for i, (img, msk) in enumerate(zip(images, masks)):
            progress(i / total, desc=f"Processing frame {i+1}/{total} (Image Background)...")
            img_uint8 = img.astype(np.uint8)
            if msk.ndim == 3:
                msk = msk[:, :, 0]
            msk = msk.astype(np.float32)

            kernel = np.ones((dilation_iterations * 2 + 1, dilation_iterations * 2 + 1), np.uint8)
            msk_uint8 = (msk > 0.5).astype(np.uint8)
            msk_dilated = cv2.dilate(msk_uint8, kernel, iterations=1)
            mask_3ch = msk_dilated[:, :, None]

            # Resize background image to match current frame dimensions
            bg_img_resized = cv2.resize(bg_img, (img.shape[1], img.shape[0]))

            result = img_uint8 * (1 - mask_3ch) + bg_img_resized * mask_3ch
            result = result.astype(np.uint8)

            output_frames.append(result)

    elif inpaint_mode == "GPU (MiniMax-Remover)":
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

            images_tensor = (torch.from_numpy(np.array(images)).float() / 127.5) - 1.0

            mask_list = []
            for m in masks:
                if m.ndim == 3:
                    m = m[:, :, 0]
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
                iterations=0
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
            progress(i / total, desc=f"Inpainting frame {i+1}/{total} ({inpaint_mode})...")
            img_uint8 = img.astype(np.uint8)
            if msk.ndim == 3:
                msk = msk[:, :, 0]
            msk = msk.astype(np.float32)

            kernel = np.ones((dilation_iterations * 2 + 1, dilation_iterations * 2 + 1), np.uint8)
            msk_uint8 = (msk > 0.5).astype(np.uint8) * 255
            msk_uint8 = msk_uint8.astype(np.uint8)
            msk_uint8 = cv2.dilate(msk_uint8, kernel, iterations=1)

            if inpaint_mode in ("Fast (OpenCV Telea)", "Fast (OpenCV NS)"):
                flag = cv2.INPAINT_TELEA if inpaint_mode == "Fast (OpenCV Telea)" else cv2.INPAINT_NS
                # img_uint8 is RGB; cv2.inpaint is channel-agnostic so result is also RGB — no conversion needed
                result = cv2.inpaint(img_uint8, msk_uint8, inpaintRadius=int(telea_radius), flags=flag)
            elif inpaint_mode == "LaMa ONNX (faster CPU)" and lama_onnx_model is not None:
                result = lama_onnx_model(img_uint8, msk_uint8, lama_config)
                result = np.clip(result, 0, 255).astype(np.uint8)
                result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            else:
                # Default: LaMa (PyTorch)
                result = lama_model(img_uint8, msk_uint8, lama_config)
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
    os.system(f'ffmpeg -y -i "{video_file_tmp}" -i "{video_path}" -c:v copy -c:a copy -map 0:v:0 -map 1:a:0? "{video_file}"')
    
    if os.path.exists(video_file_tmp):
        os.remove(video_file_tmp)
        
    progress(1.0, desc="Done!")
    return video_file


def get_video_duration(video_path):
    if video_path is None or not os.path.isfile(video_path):
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


def track_video(start_time, end_time, is_static, selection_mode, mask_data, video_state, progress=gr.Progress()):
    if video_state["origin_images"] is None:
        gr.Warning("Please extract a frame first")
        return None

    if selection_mode == "Draw Mask":
        if not mask_data:
            gr.Warning("Please draw a mask on the image!")
            return None
            
        import base64
        import io
        
        try:
            header, encoded = mask_data.split(",", 1)
            data = base64.b64decode(encoded)
            mask_pil = Image.open(io.BytesIO(data))
            mask_np = np.array(mask_pil)
            
            if mask_np.shape[-1] == 4:
                mask_2d = (mask_np[:, :, 3] > 0).astype(np.float32)
            else:
                mask_2d = (mask_np[:, :, 0] > 0).astype(np.float32)
                
            if mask_2d.sum() == 0:
                gr.Warning("Please draw a mask on the image!")
                return None
                
            height, width = video_state["origin_images"][0].shape[0:2]
            mask_2d = cv2.resize(mask_2d, (width, height))
            video_state["masks"] = np.expand_dims(mask_2d, axis=0)
        except Exception as e:
            gr.Warning("Failed to parse mask data.")
            return None
    else:
        if video_state.get("masks") is None:
            gr.Warning("Please complete target segmentation on the first frame first, then click Tracking")
            return None

    obj_id = video_state["obj_id"]

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

    color = np.array(COLOR_PALETTE[int(time.time()) % len(COLOR_PALETTE)], dtype=np.float32) / 255.0
    color = color[None, None, :]
    total_frames = len(images)
    output_frames = []
    mask_frames = []

    if is_static:
        # Object doesn't move — tile the frame-0 mask across every frame, skip SAM2 entirely
        progress(0.2, desc="Static object: tiling frame-0 mask across all frames...")
        base_mask = video_state["masks"][0]
        if base_mask.ndim == 3:
            base_mask = base_mask[:, :, 0:1]  # keep (H, W, 1)
        else:
            base_mask = base_mask[:, :, None]
        base_mask_3ch = np.repeat(base_mask, 3, axis=2).astype(np.float32)

        for idx, img in enumerate(images):
            progress(0.2 + 0.75 * (idx + 1) / total_frames,
                     desc=f"Compositing frame {idx + 1}/{total_frames}...")
            frame = img.astype(np.float32) / 255.0
            mask_frames.append(base_mask_3ch)
            painted = (1 - base_mask_3ch * 0.5) * frame + base_mask_3ch * 0.5 * color
            output_frames.append(np.uint8(np.clip(painted * 255, 0, 255)))
    else:
        images_arr = np.array(images)
        progress(0.2, desc="Initialising SAM2 tracker...")
        inference_state = video_predictor.init_state(images=images_arr / 255, device=device)
        video_state["inference_state"] = inference_state

        if len(torch.from_numpy(video_state["masks"][0]).shape) == 3:
            mask = torch.from_numpy(video_state["masks"][0])[:, :, 0]
        else:
            mask = torch.from_numpy(video_state["masks"][0])

        video_predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=obj_id,
            mask=mask
        )

        progress(0.3, desc=f"Tracking {total_frames} frames...")
        for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
            frame = images_arr[out_frame_idx].astype(np.float32) / 255.0
            mask = np.zeros((H, W, 3), dtype=np.float32)
            for i, logit in enumerate(out_mask_logits):
                out_mask = logit.cpu().squeeze().detach().numpy()
                out_mask = (out_mask[:, :, None] > 0).astype(np.float32)
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

def load_tracked_video(original_path, tracked_path, video_state, progress=gr.Progress()):
    original_path = video_state.get("video_path") or original_path
    if not original_path or not os.path.isfile(original_path):
        gr.Warning("Please upload the original video first.")
        return None, video_state
    if not tracked_path or not os.path.isfile(tracked_path):
        gr.Warning("Please upload the previously tracked video.")
        return None, video_state

    progress(0, desc="Reading original video frames...")
    vr_orig = VideoReader(original_path, ctx=cpu(0))
    fps = vr_orig.get_avg_fps()
    orig_frames = [vr_orig[i].asnumpy() for i in range(len(vr_orig))]
    del vr_orig

    progress(0.1, desc="Reading tracked video frames...")
    vr_tracked = VideoReader(tracked_path, ctx=cpu(0))
    tracked_frames = [vr_tracked[i].asnumpy() for i in range(len(vr_tracked))]
    del vr_tracked

    n = min(len(orig_frames), len(tracked_frames))
    height, width = orig_frames[0].shape[:2]

    if orig_frames[0].shape[0] > orig_frames[0].shape[1]:
        W_ = W
        H_ = int(W_ * height / width)
    else:
        H_ = H
        W_ = int(H_ * width / height)

    progress(0.2, desc="Extracting masks from diff...")
    mask_frames = []
    images_resized = []
    for i in range(n):
        progress(0.2 + 0.7 * i / n, desc=f"Processing frame {i+1}/{n}...")
        orig = cv2.resize(orig_frames[i].astype(np.uint8), (W_, H_))
        tracked = cv2.resize(tracked_frames[i].astype(np.uint8), (W_, H_))

        # Diff in grayscale — overlay pixels differ from original
        orig_gray = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY).astype(np.int16)
        tracked_gray = cv2.cvtColor(tracked, cv2.COLOR_RGB2GRAY).astype(np.int16)
        diff = np.abs(orig_gray - tracked_gray).astype(np.uint8)

        # Threshold: any pixel that changed by more than 15 counts as masked
        _, mask = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
        mask = mask.astype(np.float32) / 255.0
        mask_frames.append(mask)
        images_resized.append(orig)

    video_state["masks"] = mask_frames
    video_state["origin_images"] = images_resized
    video_state["start_frame"] = 0
    video_state["end_frame"] = n
    video_state["fps"] = fps
    video_state["original_height"] = height
    video_state["original_width"] = width

    progress(1.0, desc="Done! Click Remove to proceed.")
    gr.Info(f"Loaded {n} frames with masks extracted from tracked video. Click Remove to proceed.")
    return tracked_path, video_state


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

lama_model, lama_onnx_model, image_predictor, video_predictor = get_lama_and_predictors()

def build_tab(tab_id, tab_label, inpaint_modes, btn_label, desc_info):
    with gr.Tab(tab_label, id=tab_id):
        video_state = gr.State({
            "origin_images": None,
            "inference_state": None,
            "masks": None,
            "painted_images": None,
            "video_path": None,
            "input_points": [],
            "scaled_points": [],
            "input_labels": [],
            "frame_idx": 0,
            "obj_id": 1
        })

        with gr.Column():
            video_input = gr.Video(
                label="Upload Video",
                elem_id=f"my-video1-{tab_id}",
                height=300
            )
            
            def on_upload_copy(vp, vs):
                if not vp or not os.path.isfile(vp):
                    return gr.update(), vs
                try:
                    abs_storage = os.path.realpath(STORAGE_DIR)
                    vp_abs = os.path.realpath(vp)
                    if vp_abs.startswith(abs_storage) or "cartoon" in vp or "normal_videos" in vp:
                        vs["video_path"] = vp
                        return gr.update(), vs
                    name = os.path.basename(vp)
                    new_path = os.path.join(STORAGE_DIR, f"{int(time.time())}_{name}")
                    shutil.copy2(vp, new_path)
                    vs["video_path"] = new_path
                except Exception:
                    pass
                return gr.update(), vs

            video_input.change(fn=on_upload_copy, inputs=[video_input, video_state], outputs=[video_input, video_state])
            gr.Markdown(f"<div style='text-align:center;color:#aaa;font-size:12px'>📌 <b>How to use:</b> {desc_info}</div>", elem_id="my-btn")
            
            current_time_box = gr.Number(value=0, visible=False)
            get_info_btn = gr.Button("Extract Current Frame", elem_id="my-btn")

            gr.Examples(
                examples=[
                    ["./cartoon/0.mp4"], ["./cartoon/1.mp4"], ["./cartoon/2.mp4"], ["./cartoon/3.mp4"], ["./cartoon/4.mp4"],
                    ["./normal_videos/0.mp4"], ["./normal_videos/1.mp4"], ["./normal_videos/3.mp4"], ["./normal_videos/4.mp4"], ["./normal_videos/5.mp4"],
                ],
                inputs=[video_input],
                label="Choose a video to process.",
                elem_id="my-btn2"
            )

            with gr.Row(elem_id="my-btn"):
                selection_mode = gr.Radio(
                    ["Click Points", "Draw Mask"],
                    value="Click Points",
                    label="Selection Mode",
                    info="Choose 'Click Points' to use SAM2 AI tracking, or 'Draw Mask' to manually paint the area you want to track/replace."
                )

            image_output = gr.Image(label="First Frame Segmentation", interactive=True, elem_id="my-video", height=300)
            image_editor = gr.Image(label="Draw Mask", interactive=False, elem_classes="my-editor-class", elem_id="my-editor-video", height=300, visible=False)
            mask_data_input = gr.Textbox(visible=False, elem_id="mask-data-input")
            
            with gr.Row(elem_id="my-btn", visible=True) as clicks_row:
                point_prompt = gr.Radio(
                    ["Positive", "Negative", "Pick Color"],
                    label="Click Type",
                    value="Positive",
                    info="Positive = click ON the object you want to remove. Negative = click on areas you do NOT want selected. Pick Color = click to pick a color for Solid Color mode."
                )
                clear_btn = gr.Button("Clear All Clicks")

            def toggle_selection_mode(mode):
                return (
                    gr.update(visible=(mode == "Click Points")),
                    gr.update(visible=(mode == "Draw Mask")),
                    gr.update(visible=(mode == "Click Points"))
                )
            
            selection_mode.change(
                toggle_selection_mode,
                inputs=[selection_mode],
                outputs=[image_output, image_editor, clicks_row]
            )

            use_tracked_checkbox = gr.Checkbox(
                label="Use previously tracked video instead of tracking",
                value=False,
                elem_id="my-btn"
            )

            with gr.Column(visible=True) as tracking_section:
                timeline_ruler = gr.HTML("<div style='color:#aaa;font-size:12px;padding:6px 0'>Upload a video to see the timeline ruler.</div>", elem_id="my-btn")
                with gr.Row(elem_id="my-btn"):
                    start_time_slider = gr.Slider(minimum=0, maximum=60, value=0, step=0.5, label="Start Time (s)")
                    end_time_slider = gr.Slider(minimum=0, maximum=60, value=10, step=0.5, label="End Time (s)")
                with gr.Row(elem_id="my-btn"):
                    track_btn = gr.Button("Tracking")
                    stop_track_btn = gr.Button("Stop Tracking", variant="stop")
                    static_object_checkbox = gr.Checkbox(label="Static Object", value=False)

            with gr.Column(visible=False) as upload_tracked_section:
                tracked_video_upload = gr.Video(label="Previously Tracked Video", elem_id="my-video", height=200)
                load_tracked_btn = gr.Button("Load Tracked Video", elem_id="my-btn")

            video_output = gr.Video(label="Tracking Result", elem_id="my-video", height=300)

            def toggle_tracking_mode(use_tracked):
                return gr.update(visible=not use_tracked), gr.update(visible=use_tracked)

            use_tracked_checkbox.change(
                toggle_tracking_mode,
                inputs=[use_tracked_checkbox],
                outputs=[tracking_section, upload_tracked_section]
            )

            with gr.Column(elem_id="my-btn"):
                dilation_slider = gr.Slider(
                    minimum=1, maximum=20, value=6, step=1,
                    label="Mask Dilation",
                    info="Expands the removal mask outward by this many pixels before inpainting. Recommended: 4–8."
                )

            with gr.Row(elem_id="my-btn"):
                inpaint_mode_dropdown = gr.Dropdown(
                    choices=inpaint_modes,
                    value=inpaint_modes[0],
                    label="Processing Mode"
                )
            
            with gr.Column(elem_id="my-btn", visible=False) as telea_options:
                telea_radius_slider = gr.Slider(minimum=1, maximum=40, value=15, step=1, label="OpenCV Inpaint Radius")
            with gr.Column(elem_id="my-btn", visible=False) as color_options:
                mask_color_picker = gr.ColorPicker(label="Mask Color", value="#000000")
            with gr.Column(elem_id="my-btn", visible=False) as bg_video_options:
                gr.Markdown("<div style='color:#aaa;font-size:12px;padding:6px 0'>Upload a video to replace the masked object/background.</div>")
                bg_video_upload = gr.Video(label="Background Video", elem_id="my-video", height=300)
            with gr.Column(elem_id="my-btn", visible=False) as bg_image_options:
                gr.Markdown("<div style='color:#aaa;font-size:12px;padding:6px 0'>Upload an image to replace the masked object/background.</div>")
                bg_image_upload = gr.Image(label="Background Image", type="filepath", elem_id="my-video", height=300)

            def toggle_options(mode):
                return (
                    gr.update(visible=(mode in ("Fast (OpenCV Telea)", "Fast (OpenCV NS)"))),
                    gr.update(visible=(mode == "Solid Color")),
                    gr.update(visible=(mode == "Video Background")),
                    gr.update(visible=(mode == "Image Background"))
                )

            inpaint_mode_dropdown.change(
                toggle_options,
                inputs=[inpaint_mode_dropdown],
                outputs=[telea_options, color_options, bg_video_options, bg_image_options]
            )

            with gr.Row(elem_id="my-btn"):
                remove_btn = gr.Button(btn_label)
                stop_remove_btn = gr.Button("Stop Process", variant="stop")
            remove_video = gr.Video(label="Output Results", elem_id="my-video", height=300)
            
            with gr.Row(elem_id="my-btn"):
                download_btn = gr.DownloadButton("Download Output Video", interactive=False)
            
            def enable_download(video):
                if video:
                    return gr.update(value=video, interactive=True)
                return gr.update(interactive=False)

            remove_event = remove_btn.click(
                inference_and_return_video,
                inputs=[dilation_slider, inpaint_mode_dropdown, telea_radius_slider, mask_color_picker, bg_video_upload, bg_image_upload, video_state],
                outputs=remove_video
            ).then(
                fn=enable_download,
                inputs=[remove_video],
                outputs=[download_btn]
            )
            stop_remove_btn.click(fn=None, cancels=[remove_event])
            
            get_info_btn.click(
                fn=get_video_info,
                inputs=[video_input, current_time_box, video_state],
                outputs=[image_output, image_editor],
                js=f"(vp, t, vs) => [vp, (document.querySelector('#my-video1-{tab_id} video')?.currentTime ?? 0), vs]"
            )
            video_input.change(
                get_video_duration,
                inputs=[video_input],
                outputs=[start_time_slider, end_time_slider, timeline_ruler]
            )
            image_output.select(fn=segment_frame, inputs=[point_prompt, video_state, mask_color_picker], outputs=[image_output, mask_color_picker])
            clear_btn.click(clear_clicks, inputs=video_state, outputs=image_output)
            track_event = track_btn.click(track_video, inputs=[start_time_slider, end_time_slider, static_object_checkbox, selection_mode, mask_data_input, video_state], outputs=video_output)
            stop_track_btn.click(fn=None, cancels=[track_event])
            load_tracked_btn.click(
                load_tracked_video,
                inputs=[video_input, tracked_video_upload, video_state],
                outputs=[video_output, video_state]
            )

        return {
            "video_state": video_state,
            "video_input": video_input,
            "image_output": image_output,
            "image_editor": image_editor,
            "mask_data_input": mask_data_input,
            "video_output": video_output,
            "tracked_video_upload": tracked_video_upload,
            "remove_video": remove_video,
            "start_time_slider": start_time_slider,
            "end_time_slider": end_time_slider,
            "timeline_ruler": timeline_ruler,
            "download_btn": download_btn,
            "static_object_checkbox": static_object_checkbox,
            "use_tracked_checkbox": use_tracked_checkbox,
            "tracking_section": tracking_section,
            "upload_tracked_section": upload_tracked_section,
        }

with gr.Blocks() as demo:
    gr.Markdown(f"<div style='text-align:center;'>{text}</div>")
    
    reset_btn = gr.Button("Reset Everything", elem_id="reset-btn", variant="stop")

    demo.css = """
    #my-btn { width: 60% !important; margin: 0 auto; }
    [id^='my-video1'] { width: 60% !important; max-height: 280px !important; margin: 0 auto; }
    [id^='my-video1'] video { max-height: 240px !important; object-fit: contain !important; }
    #my-video { width: 60% !important; max-height: 320px !important; margin: 0 auto; }
    #my-video video, #my-video img { max-height: 260px !important; object-fit: contain !important; }
    #my-md { margin: 0 auto; }
    #my-btn2 { width: 60% !important; margin: 0 auto; }
    #reset-btn { width: 60% !important; margin: 8px auto !important; display: block !important; }
    #my-btn2 button { width: 120px !important; max-width: 120px !important; min-width: 120px !important; height: 70px !important; max-height: 70px !important; min-height: 70px !important; margin: 8px !important; border-radius: 8px !important; overflow: hidden !important; white-space: normal !important; }
    video::-webkit-media-controls-timeline { height: 10px !important; padding: 0 !important; }
    .video-container input[type=range], video ~ * input[type=range] { height: 10px !important; cursor: pointer !important; }
    input[type=range]::-webkit-slider-runnable-track { height: 8px !important; border-radius: 4px !important; }
    input[type=range]::-webkit-slider-thumb { width: 10px !important; height: 28px !important; margin-top: -10px !important; border-radius: 3px !important; background: white !important; border: 1px solid #ccc !important; box-shadow: 0 1px 4px rgba(0,0,0,0.4) !important; cursor: pointer !important; }
    input[type=range]::-moz-range-thumb { width: 10px !important; height: 28px !important; border-radius: 3px !important; background: white !important; border: 1px solid #ccc !important; box-shadow: 0 1px 4px rgba(0,0,0,0.4) !important; cursor: pointer !important; }
    """

    with gr.Tabs():
        tab1_components = build_tab(
            tab_id="tab1",
            tab_label="Remove Object",
            inpaint_modes=["LaMa (PyTorch)", "LaMa ONNX (faster CPU)", "Fast (OpenCV Telea)", "Fast (OpenCV NS)", "GPU (MiniMax-Remover)"],
            btn_label="Remove Object",
            desc_info="Upload a video → pause at the frame with the object → click <i>Extract Current Frame</i> → click the object in the image → adjust Start/End time → click <i>Tracking</i> → click <i>Remove Object</i>"
        )
        
        tab2_components = build_tab(
            tab_id="tab2",
            tab_label="Replace Background",
            inpaint_modes=["Solid Color", "Video Background", "Image Background"],
            btn_label="Replace Background",
            desc_info="Upload a video → pause at the frame with the background → click <i>Extract Current Frame</i> → click the background in the image → click <i>Tracking</i> → choose mode and click <i>Replace Background</i>"
        )

    def reset_all():
        fresh_state = {"origin_images": None, "inference_state": None, "masks": None, "painted_images": None, "video_path": None, "input_points": [], "scaled_points": [], "input_labels": [], "frame_idx": 0, "obj_id": 1}
        return (
            fresh_state, None, None, None, "", None, None, None, gr.update(value=0, maximum=60), gr.update(value=10, maximum=60), "<div style='color:#aaa;font-size:12px;padding:6px 0'>Upload a video to see the timeline ruler.</div>", gr.update(interactive=False), False, False, gr.update(visible=True), gr.update(visible=False),
            fresh_state, None, None, None, "", None, None, None, gr.update(value=0, maximum=60), gr.update(value=10, maximum=60), "<div style='color:#aaa;font-size:12px;padding:6px 0'>Upload a video to see the timeline ruler.</div>", gr.update(interactive=False), False, False, gr.update(visible=True), gr.update(visible=False)
        )

    reset_btn.click(
        fn=reset_all,
        inputs=[],
        outputs=[
            tab1_components["video_state"], tab1_components["video_input"], tab1_components["image_output"], tab1_components["image_editor"], tab1_components["mask_data_input"], tab1_components["video_output"], tab1_components["tracked_video_upload"], tab1_components["remove_video"], tab1_components["start_time_slider"], tab1_components["end_time_slider"], tab1_components["timeline_ruler"], tab1_components["download_btn"], tab1_components["static_object_checkbox"], tab1_components["use_tracked_checkbox"], tab1_components["tracking_section"], tab1_components["upload_tracked_section"],
            tab2_components["video_state"], tab2_components["video_input"], tab2_components["image_output"], tab2_components["image_editor"], tab2_components["mask_data_input"], tab2_components["video_output"], tab2_components["tracked_video_upload"], tab2_components["remove_video"], tab2_components["start_time_slider"], tab2_components["end_time_slider"], tab2_components["timeline_ruler"], tab2_components["download_btn"], tab2_components["static_object_checkbox"], tab2_components["use_tracked_checkbox"], tab2_components["tracking_section"], tab2_components["upload_tracked_section"]
        ],
        js="() => { localStorage.removeItem('minimax_tr'); localStorage.removeItem('minimax_rm'); }"
    )

    gr.HTML("""
    <script>
    (function() {
        function attachZoom(el) {
            if (el._zoomAttached) return;
            el._zoomAttached = true;
            el._zoomScale = 1;
            el._zoomOriginX = 50;
            el._zoomOriginY = 50;
            el.style.transition = 'transform 0.08s ease';
            el.style.cursor = 'zoom-in';

            const parent = el.closest('.wrap, .image-container, figure, .video-container') || el.parentElement;
            if (parent) parent.style.overflow = 'hidden';

            el.addEventListener('wheel', function(e) {
                e.preventDefault();
                e.stopPropagation();

                const rect = el.getBoundingClientRect();
                const mouseX = ((e.clientX - rect.left) / rect.width) * 100;
                const mouseY = ((e.clientY - rect.top) / rect.height) * 100;

                const isPinch = e.ctrlKey;
                const factor = e.deltaY < 0 ? (isPinch ? 1.04 : 1.12) : (isPinch ? 0.96 : 0.89);
                const next = Math.min(Math.max(el._zoomScale * factor, 1), 8);

                if (next > el._zoomScale) {
                    el._zoomOriginX = mouseX;
                    el._zoomOriginY = mouseY;
                }

                el._zoomScale = next;
                el.style.transformOrigin = el._zoomOriginX + '% ' + el._zoomOriginY + '%';
                el.style.transform = next === 1 ? '' : 'scale(' + next + ')';
                el.style.cursor = next > 1 ? 'zoom-out' : 'zoom-in';
            }, { passive: false });

            el.addEventListener('dblclick', function() {
                el._zoomScale = 1;
                el.style.transform = '';
                el.style.transformOrigin = '50% 50%';
                el.style.cursor = 'zoom-in';
            });
        }

        function scanAndAttach() {
            document.querySelectorAll('video, .gradio-image img, .image-container img, [data-testid="image"] img, img.svelte-image').forEach(el => {
                if(el.closest('.my-editor-class') || el.tagName.toLowerCase() === 'canvas') return;
                attachZoom(el);
            });
        }

        scanAndAttach();
        new MutationObserver(scanAndAttach).observe(document.body, { childList: true, subtree: true });
        
        // Polygon Drawing Logic
        let pts = [];
        let canvas = null;
        let ctx = null;
        let maskCanvas = null;
        let maskCtx = null;
        
        function setupEditor() {
            document.querySelectorAll('#my-editor-video').forEach(editorContainer => {
                if(editorContainer.dataset.polygonSetup) return;
                
                const img = editorContainer.querySelector('img');
                if(!img || !img.complete || img.naturalWidth === 0) return; // Wait for image to load
                
                editorContainer.dataset.polygonSetup = "true";
                
                const drawCanvas = document.createElement('canvas');
                drawCanvas.style.position = 'absolute';
                drawCanvas.style.top = '0';
                drawCanvas.style.left = '0';
                drawCanvas.style.width = '100%';
                drawCanvas.style.height = '100%';
                drawCanvas.style.pointerEvents = 'auto'; 
                drawCanvas.style.cursor = 'crosshair';
                drawCanvas.style.zIndex = '1000';
                
                const mCanvas = document.createElement('canvas');
                const mCtx = mCanvas.getContext('2d');
                
                editorContainer.style.position = 'relative';
                const imgParent = img.parentElement;
                imgParent.style.position = 'relative';
                imgParent.appendChild(drawCanvas);
                
                const drawCtx = drawCanvas.getContext('2d');
                
                drawCanvas.addEventListener('mousedown', (e) => {
                    if(drawCanvas.width !== img.naturalWidth) {
                        drawCanvas.width = img.naturalWidth;
                        drawCanvas.height = img.naturalHeight;
                        mCanvas.width = img.naturalWidth;
                        mCanvas.height = img.naturalHeight;
                        drawCanvas.dataset.imgSrc = img.src;
                    }
                    
                    const rect = drawCanvas.getBoundingClientRect();
                    const scaleX = drawCanvas.width / rect.width;
                    const scaleY = drawCanvas.height / rect.height;
                    
                    const x = (e.clientX - rect.left) * scaleX;
                    const y = (e.clientY - rect.top) * scaleY;
                    
                    if(pts.length > 2) {
                        const dx = x - pts[0].x;
                        const dy = y - pts[0].y;
                        if(Math.sqrt(dx*dx + dy*dy) < 20 * scaleX) {
                            pts.push({...pts[0]});
                            
                            mCtx.fillStyle = 'rgba(0, 255, 0, 0.5)';
                            mCtx.beginPath();
                            mCtx.moveTo(pts[0].x, pts[0].y);
                            for(let i=1; i<pts.length; i++) mCtx.lineTo(pts[i].x, pts[i].y);
                            mCtx.closePath();
                            mCtx.fill();
                            
                            const b64 = mCanvas.toDataURL('image/png');
                            // Find the corresponding hidden input
                            // We need to find the correct text area
                            const root = editorContainer.closest('.wrap') || document;
                            const inputs = document.querySelectorAll('#mask-data-input textarea');
                            // We might have two tabs, so we find the one currently visible or just set both
                            inputs.forEach(input => {
                                input.value = b64;
                                input.dispatchEvent(new Event('input', {bubbles: true}));
                            });
                            
                            pts = [];
                            draw();
                            return;
                        }
                    }
                    
                    pts.push({x, y});
                    draw();
                });
                
                drawCanvas.addEventListener('mousemove', (e) => {
                    if(pts.length === 0) return;
                    const rect = drawCanvas.getBoundingClientRect();
                    const scaleX = drawCanvas.width / rect.width;
                    const scaleY = drawCanvas.height / rect.height;
                    const x = (e.clientX - rect.left) * scaleX;
                    const y = (e.clientY - rect.top) * scaleY;
                    draw(x, y);
                });
                
                function draw(mouseX, mouseY) {
                    drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
                    drawCtx.drawImage(mCanvas, 0, 0);
                    
                    if(pts.length > 0) {
                        drawCtx.setLineDash([5, 5]);
                        drawCtx.lineWidth = 2 * (drawCanvas.width / drawCanvas.getBoundingClientRect().width);
                        drawCtx.strokeStyle = '#00FF00';
                        drawCtx.beginPath();
                        drawCtx.moveTo(pts[0].x, pts[0].y);
                        for(let i=1; i<pts.length; i++) {
                            drawCtx.lineTo(pts[i].x, pts[i].y);
                        }
                        if(mouseX !== undefined) {
                            drawCtx.lineTo(mouseX, mouseY);
                        }
                        drawCtx.stroke();
                        
                        drawCtx.setLineDash([]);
                        for(let p of pts) {
                            drawCtx.beginPath();
                            drawCtx.arc(p.x, p.y, 4 * (drawCanvas.width / drawCanvas.getBoundingClientRect().width), 0, Math.PI*2);
                            drawCtx.fillStyle = '#00FF00';
                            drawCtx.fill();
                        }
                    }
                }
                
                // Monitor image changes
                const observer = new MutationObserver(() => {
                    if (img.src && drawCanvas.dataset.imgSrc !== img.src) {
                        drawCanvas.dataset.imgSrc = img.src;
                        drawCanvas.width = img.naturalWidth;
                        drawCanvas.height = img.naturalHeight;
                        mCanvas.width = img.naturalWidth;
                        mCanvas.height = img.naturalHeight;
                        drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
                        mCtx.clearRect(0, 0, mCanvas.width, mCanvas.height);
                        pts = [];
                        const inputs = document.querySelectorAll('#mask-data-input textarea');
                        inputs.forEach(input => {
                            input.value = '';
                            input.dispatchEvent(new Event('input', {bubbles: true}));
                        });
                    }
                });
                observer.observe(img, { attributes: true, attributeFilter: ['src'] });
            });
        }
        
        setInterval(setupEditor, 500);
    })();
    </script>
    """)

demo.launch(server_name="0.0.0.0", server_port=8000, allowed_paths=["/tmp", os.path.abspath(STORAGE_DIR)], share=True)