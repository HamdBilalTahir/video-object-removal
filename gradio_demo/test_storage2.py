import gradio as gr
import os

STORAGE_DIR = "./storage"
os.makedirs(STORAGE_DIR, exist_ok=True)

js_load = """
async () => {
    try {
        const up = JSON.parse(localStorage.getItem('minimax_up'));
        const tr = JSON.parse(localStorage.getItem('minimax_tr'));
        return [up || null, tr || null];
    } catch(e) {
        return [null, null];
    }
}
"""

with gr.Blocks() as demo:
    with gr.Row():
        vid_up = gr.Video(label="Upload")
        vid_tr = gr.Video(label="Tracked")
        
    demo.load(fn=None, inputs=[], outputs=[vid_up, vid_tr], js=js_load)
    
    vid_up.change(fn=None, inputs=[vid_up], outputs=[], js="(val) => { if (val) localStorage.setItem('minimax_up', JSON.stringify(val)); else localStorage.removeItem('minimax_up'); }")
    vid_tr.change(fn=None, inputs=[vid_tr], outputs=[], js="(val) => { if (val) localStorage.setItem('minimax_tr', JSON.stringify(val)); else localStorage.removeItem('minimax_tr'); }")
    
    # Tracked receives input from python
    def do_track(vid):
        return vid
    
    btn = gr.Button("Track")
    btn.click(fn=do_track, inputs=[vid_up], outputs=[vid_tr])

demo.launch(server_name="0.0.0.0", server_port=8001, allowed_paths=[STORAGE_DIR])
