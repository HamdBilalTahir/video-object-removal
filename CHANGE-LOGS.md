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
