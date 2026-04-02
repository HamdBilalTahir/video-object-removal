import gradio as gr
import shutil, os, time

STORAGE_DIR = "./storage"

def on_upload_copy(vp):
    print("on_upload_copy called with vp=", vp)
    if not vp:
        return None
    abs_storage = os.path.abspath(STORAGE_DIR)
    vp_abs = os.path.abspath(vp)
    if vp_abs.startswith(abs_storage) or "cartoon" in vp or "normal_videos" in vp:
        return vp
    name = os.path.basename(vp)
    new_path = os.path.join(STORAGE_DIR, f"{int(time.time())}_{name}")
    shutil.copy2(vp, new_path)
    print("Copied to", new_path)
    return new_path

with gr.Blocks() as demo:
    v = gr.Video()
    v.upload(on_upload_copy, inputs=[v], outputs=[v])

if __name__ == "__main__":
    demo.launch(server_port=8001)
