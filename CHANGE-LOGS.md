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

### 🐛 Fixes

---

> ### Resolve OpenCV datatype crash and Gradio upload loop
>
> - **What changed:** Fixed OpenCV CV_64F crashes by converting the removal mask to uint8/float32, resolved the infinite loop during video upload in Gradio, updated requirements to include `iopaint`, and added a `.gitignore` for models.
> - **Why:** Enables stable LaMa removal, prevents the UI from freezing when updating the storage path, and provides seamless setup instructions.
> - **Files:**
>   - `gradio_demo/test.py`
>   - `requirements.txt`
>   - `README.md`
>   - `.gitignore`
