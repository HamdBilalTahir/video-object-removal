import gradio as gr
import os
import shutil

STORAGE_DIR = "./storage"
os.makedirs(STORAGE_DIR, exist_ok=True)

def on_upload(video_path):
    if not video_path:
        return None, None
    print("Uploaded:", video_path)
    new_path = os.path.join(STORAGE_DIR, os.path.basename(video_path))
    if not os.path.exists(new_path) and video_path != new_path:
        shutil.copy2(video_path, new_path)
    return new_path, new_path

def clear_storage():
    print("Clearing storage")
    return None, None, None, None, None, None

js_load = """
async () => {
    const up = localStorage.getItem('minimax_up');
    const tr = localStorage.getItem('minimax_tr');
    const rm = localStorage.getItem('minimax_rm');
    return [up || null, tr || null, rm || null];
}
"""

js_save_up = "(val) => { if (val) localStorage.setItem('minimax_up', val); else localStorage.removeItem('minimax_up'); return val; }"

with gr.Blocks() as demo:
    with gr.Row():
        vid_up = gr.Video(label="Upload")
        vid_tr = gr.Video(label="Tracked")
        vid_rm = gr.Video(label="Removed")
        
        path_up = gr.Textbox(visible=False)
        path_tr = gr.Textbox(visible=False)
        path_rm = gr.Textbox(visible=False)
        
    demo.load(fn=None, inputs=[], outputs=[path_up, path_tr, path_rm], js=js_load)
    
    # When page loads, update videos from textboxes
    path_up.change(fn=lambda x: x, inputs=[path_up], outputs=[vid_up])
    path_tr.change(fn=lambda x: x, inputs=[path_tr], outputs=[vid_tr])
    path_rm.change(fn=lambda x: x, inputs=[path_rm], outputs=[vid_rm])
    
    # When user uploads, copy to storage and update textbox (which updates JS)
    vid_up.upload(fn=on_upload, inputs=[vid_up], outputs=[vid_up, path_up])
    
    path_up.change(fn=None, inputs=[path_up], outputs=[], js=js_save_up)
    
    # When user clears upload, clear all
    vid_up.clear(fn=clear_storage, inputs=[], outputs=[vid_up, vid_tr, vid_rm, path_up, path_tr, path_rm])

demo.launch(server_name="0.0.0.0", server_port=8001, allowed_paths=[STORAGE_DIR])
