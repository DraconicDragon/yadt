import gradio as gr

def create_threshold_options(args, include_mcut_checkboxes=True):
    """
    Creates and returns Gradio sliders and optionally checkboxes for general and character thresholds.
    """
    with gr.Group():
        with gr.Row():
            general_thresh = gr.Slider(
                0,
                1,
                step=args.score_slider_step,
                value=args.score_general_threshold,
                label="General Tags Threshold",
            )
            character_thresh = gr.Slider(
                0,
                1,
                step=args.score_slider_step,
                value=args.score_character_threshold,
                label="Character Tags Threshold",
            )
        
        if include_mcut_checkboxes:
            with gr.Row():
                general_mcut_enabled = gr.Checkbox(
                    value=False,
                    label="Use MCut threshold for General Tags",
                )
                character_mcut_enabled = gr.Checkbox(
                    value=False,
                    label="Use MCut threshold for Character Tags",
                )
        else:
            general_mcut_enabled = False
            character_mcut_enabled = False

    return general_thresh, general_mcut_enabled, character_thresh, character_mcut_enabled