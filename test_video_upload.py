import gradio as gr

def check_video(v):
    print("Video type:", type(v), "value:", v)
    return v

with gr.Blocks() as demo:
    v = gr.Video()
    v.upload(check_video, v, v)

if __name__ == "__main__":
    check_video("test")
