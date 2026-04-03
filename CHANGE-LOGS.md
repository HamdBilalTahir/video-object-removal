## 🗓️ **2026-04-03**

---

### 🐛 Fixes

---


> ### Fix OpenCV Telea inpainting changing video colors
>
> - **What changed:** Removed the incorrect `cv2.cvtColor(result, cv2.COLOR_BGR2RGB)` call after `cv2.inpaint` in the Telea branch.
> - **Why:** `img_uint8` is already RGB. `cv2.inpaint` is channel-agnostic and preserves the input channel order, so the result is also RGB. The extra conversion was swapping red and blue channels across the entire output video.
> - **Files:**
>   - `gradio_demo/test.py`

> ### Replace tracked video accordion with checkbox toggle
>
> - **What changed:** Replaced the `gr.Accordion` for the tracked video upload with a checkbox ("Use previously tracked video instead of tracking"). Checking it hides the tracking section (sliders, Track button, Static Object checkbox) and shows the upload + Load button. Unchecked by default.
> - **Why:** The accordion always showed both options at once, which was confusing. The checkbox makes it a clear either/or choice.
> - **Files:**
>   - `gradio_demo/test.py`

### ✨ Features

---

> ### Add OpenCV NS inpainting mode
>
> - **What changed:** Added "Fast (OpenCV NS)" as a fifth inpainting option alongside Telea. NS (Navier-Stokes) uses fluid dynamics equations to fill the masked region, producing better edge continuity than Telea for larger objects. Both Telea and NS share the same inpaint radius slider.
> - **Why:** Telea smears on larger mask regions. NS handles them better while remaining instant (no model loading).
> - **Files:**
>   - `gradio_demo/test.py`

> ### Add OpenCV inpaint radius slider with optimal default
>
> - **What changed:** Added an "OpenCV Inpaint Radius" slider (range 1–40, default 15) that appears when either OpenCV Telea or NS is selected. Controls how far the algorithm samples outward from the mask boundary. Default set to 15 for real-world object sizes.
> - **Why:** The previous hardcoded value of 5 produced visible smearing on anything larger than a small scratch. LaMa and MiniMax-Remover are neural models with no equivalent parameter, so the slider is OpenCV-only.
> - **Files:**
>   - `gradio_demo/test.py`

> ### Add per-mode scenario guidance to inpainting dropdown
>
> - **What changed:** Updated the dropdown info text to describe the best use case for each mode: LaMa PyTorch (any background, best CPU quality), LaMa ONNX (same quality, ~2-4x faster on CPU), OpenCV Telea (thin/small objects on uniform backgrounds, instant), OpenCV NS (larger objects on uniform backgrounds, instant), GPU MiniMax-Remover (highest quality overall, requires CUDA).
> - **Files:**
>   - `gradio_demo/test.py`

> ### Upload previously tracked video to skip re-tracking
>
> - **What changed:** Added a "Upload Previously Tracked Video" input and a "Load Tracked Video" button below the Track button. When clicked, the function diffs the original and tracked video frame-by-frame, thresholds pixel differences (>15 intensity units) to extract binary masks, and populates `video_state` with frames and masks — allowing the user to click Remove immediately without re-running SAM2.
> - **Why:** When a session is killed mid-way, the user previously had to re-track from scratch even if they already had the tracking output saved. Now they can resume from the tracked video directly.
> - **Files:**
>   - `gradio_demo/test.py`

> ### Inpainting mode dropdown (LaMa / LaMa ONNX / OpenCV Telea / GPU)
>
> - **What changed:** Replaced the "Use GPU" checkbox with a dropdown offering four inpainting backends: LaMa PyTorch (default), LaMa ONNX (faster CPU), OpenCV Telea (near-instant, good for uniform backgrounds), and GPU MiniMax-Remover.
> - **Why:** Gives users a speed/quality tradeoff choice without needing to know the internals. OpenCV Telea is ~100x faster than LaMa for simple backgrounds like solid walls.
> - **Files:**
>   - `gradio_demo/test.py`

> ### Static Object mode — skip SAM2 tracking entirely
>
> - **What changed:** Added a "Static Object" checkbox next to the Track button. When checked, the frame-0 mask is tiled across all frames without running SAM2 video propagation.
> - **Why:** For objects that don't move (e.g. a mic on a fixed camera shot), SAM2 tracking is unnecessary and can take several minutes. Static mode reduces tracking time to near-zero.
> - **Files:**
>   - `gradio_demo/test.py`

> ### Reset button to clear all session state
>
> - **What changed:** Added a red "Reset" button below the title that clears all components (upload video, segmentation image, tracking video, removal video, sliders, download button) and resets `video_state` to its initial empty values. Also clears localStorage keys.
> - **Why:** Allows users to start a completely fresh session without reloading the page.
> - **Files:**
>   - `gradio_demo/test.py`

### 🐛 Fixes

---

> ### Fix Gradio ValueError on corrupted or missing file paths
>
> - **What changed:** Added `os.path.isfile()` guards in `on_upload_copy`, `get_video_info`, and `get_video_duration`. Removed localStorage save/restore for all video components. `on_upload_copy` now always returns `gr.update()` instead of the storage path.
> - **Why:** Gradio validates every file path against its internal temp dir (`/tmp/gradio`). Paths in `STORAGE_DIR` and stale localStorage paths fail this check before any Python code runs, causing an unhandled `ValueError`. Keeping the component pointing to Gradio's managed temp path avoids the check entirely.
> - **Files:**
>   - `gradio_demo/test.py`

> ### Fix output video not showing after removal (Gradio temp path cleaned up)
>
> - **What changed:** `on_upload_copy` now stores the durable storage copy path into `video_state["video_path"]`. `get_video_info` prefers `video_state["video_path"]` over the Gradio temp path argument. ffmpeg paths are now quoted.
> - **Why:** Gradio's temp dir is cleaned during long operations (5+ minute tracking runs). `video_state["video_path"]` was still pointing to the temp path, causing ffmpeg audio mux to fail with "No such file or directory" and returning no output video.
> - **Files:**
>   - `gradio_demo/test.py`

---

## 🗓️ **2026-04-02**

---

### 📚 Docs

---

> ### Update Gradio startup commands in README
>
> - **What changed:** Updated the README to include activating the virtual environment and using `gradio test.py` with relative paths.
> - **Why:** Provides clearer instructions for users to run the local Gradio demo properly using a virtual environment.
> - **Files:**
>   - `README.md`

### 🔧 DevOps / Build

---

> ### Add root SAM2-Video-Predictor to gitignore
>
> - **What changed:** Added `SAM2-Video-Predictor/` to the root `.gitignore` file.
> - **Why:** Prevents accidentally committing the large downloaded SAM2 model checkpoints from the root directory.
> - **Files:**
>   - `.gitignore`

### ✨ Features

---

> ### Preserve original audio and video quality in output
>
> - **What changed:** Updated `write_videofile` to use `-crf 18` for visually lossless video encoding and added an `ffmpeg` multiplexing step to directly copy (`-c:a copy`) the original audio stream into the final output.
> - **Why:** Guarantees that the audio quality is 100% mathematically identical to the original video, and the video quality is visually indistinguishable.
> - **Files:**
>   - `gradio_demo/test.py`

### 💅 Styling and UI Improvements

---

> ### Improve slider thumb visibility
>
> - **What changed:** Replaced the circular slider thumb with a taller vertical bar style pointer.
> - **Why:** Makes the timeline tracker and other sliders significantly easier to grab and drag accurately on both mobile and desktop.
> - **Files:**
>   - `gradio_demo/test.py`

### ✨ Features

---

> ### Enable public share link for Gradio
>
> - **What changed:** Added `share=True` to the Gradio `launch()` function.
> - **Why:** Allows the local app to generate a public link that can be shared instantly across devices over the internet.
> - **Files:**
>   - `gradio_demo/test.py`

### 🐛 Fixes

---

> ### Fix hydra KeyError on Gradio reload
>
> - **What changed:** Added a sys.modules check to mock `__main__` in `sam2/__init__.py`.
> - **Why:** Solves a known bug where Gradio's hot reloader clears `__main__` and causes hydra to crash on initialization.
> - **Files:**
>   - `gradio_demo/sam2/__init__.py`

> ### Break video upload infinite loop
>
> - **What changed:** Updated `on_upload_copy` in the Gradio UI to return `gr.update()` instead of the video path when the uploaded video is already in the storage directory or is an example video.
> - **Why:** Prevents Gradio from repeatedly triggering the `change` event infinitely when a file is mapped to the same component output.
> - **Files:**
>   - `gradio_demo/test.py`

> ### Fix diffusers dependency conflict
>
> - **What changed:** Removed strict `0.33.1` version pinning for `diffusers` in `requirements.txt`.
> - **Why:** Resolves a Pip resolution conflict where `iopaint` requires `diffusers==0.27.2`.
> - **Files:**
>   - `requirements.txt`

> ### Resolve OpenCV datatype crash and Gradio upload loop
>
> - **What changed:** Fixed OpenCV CV_64F crashes by converting the removal mask to uint8/float32, resolved the infinite loop during video upload in Gradio, updated requirements to include `iopaint`, and added a `.gitignore` for models.
> - **Why:** Enables stable LaMa removal, prevents the UI from freezing when updating the storage path, and provides seamless setup instructions.
> - **Files:**
>   - `gradio_demo/test.py`
>   - `requirements.txt`
>   - `README.md`
>   - `.gitignore`
