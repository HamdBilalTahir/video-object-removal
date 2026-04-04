import gradio as gr

def process(img):
    print(type(img))
    if isinstance(img, dict):
        print(img.keys())
        if "layers" in img:
            print(f"Layers count: {len(img['layers'])}")
            if len(img['layers']) > 0:
                print(img['layers'][0].shape)
    return img

with gr.Blocks() as demo:
    editor = gr.ImageEditor(type="numpy")
    btn = gr.Button("Submit")
    btn.click(process, inputs=editor, outputs=None)

demo.launch()
