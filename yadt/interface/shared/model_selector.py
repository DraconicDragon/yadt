import gradio as gr

from yadt import tagger_shared


def create_model_selector():
    """
    Creates and returns Gradio components for model selection.
    """
    with gr.Group():
        model_repo = gr.Dropdown(
            tagger_shared.dropdown_list,
            value=tagger_shared.default_repo,
            label="Model",
        )
        with gr.Row():
            custom_model = gr.Textbox(
                value="",
                show_label=False,
                placeholder="Custom model path (may not work)",
                scale=3,
            )
            use_custom_model = gr.Checkbox(
                value=False,
                label="Use Custom Model",
                scale=1,
            )
    return model_repo, custom_model, use_custom_model
