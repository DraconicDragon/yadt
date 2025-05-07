import gradio as gr

from PIL import Image

from yadt import tagger_shared
from yadt import process_prediction
from yadt.interface import ui_utils
from yadt.interface.shared.model_selector import create_model_selector
from yadt.interface.shared.wd_tagger_threshold import create_threshold_options


def predict(args):
    @ui_utils.gradio_error
    def _predict(
        image: Image,
        model_repo: str,
        custom_model: str,
        use_custom_model: bool,
        general_thresh: float,
        general_mcut_enabled: bool,
        character_thresh: float,
        character_mcut_enabled: bool,
        replace_underscores: bool,
        trim_general_tag_dupes: bool,
        escape_brackets: bool,
    ):
        assert image is not None, "No image selected"

        if use_custom_model:
            model_repo = custom_model

        tagger_shared.predictor.load_model(model_repo, is_custom_model=use_custom_model, device=args.device)

        return process_prediction.post_process_prediction(
            *tagger_shared.predictor.predict(image),
            general_thresh,
            general_mcut_enabled,
            character_thresh,
            character_mcut_enabled,
            replace_underscores,
            trim_general_tag_dupes,
            escape_brackets,
        )

    return _predict


def ui(args):
    with gr.Row():
        with gr.Column(variant="panel"):
            image = gr.Image(type="pil", image_mode="RGBA", label="Input")

            model_repo, custom_model, use_custom_model = create_model_selector()

            (
                general_thresh,
                general_mcut_enabled,
                character_thresh,
                character_mcut_enabled,
            ) = create_threshold_options(args, include_mcut_checkboxes=True)

            with gr.Row():
                replace_underscores = gr.Checkbox(
                    value=True,
                    label="Replace underscores with spaces",
                    scale=1,
                )
                trim_general_tag_dupes = gr.Checkbox(
                    value=True,
                    label="Trim duplicate general tags",
                    scale=1,
                )
                escape_brackets = gr.Checkbox(
                    value=True,
                    label="Escape brackets (for webui)",
                    scale=1,
                )
            with gr.Row():
                clear = gr.ClearButton(
                    components=[
                        image,
                        model_repo,
                        custom_model,
                        use_custom_model,
                        general_thresh,
                        general_mcut_enabled,
                        character_thresh,
                        character_mcut_enabled,
                    ],
                    variant="secondary",
                    size="lg",
                )

                submit = gr.Button(value="Submit", variant="primary", size="lg")

        with gr.Column(variant="panel"):
            sorted_general_strings = gr.Textbox(label="Output (string)")
            rating = gr.Label(label="Rating")
            character_res = gr.Label(label="Output (characters)")
            general_res = gr.Label(label="Output (tags)")
            clear.add(
                [
                    sorted_general_strings,
                    rating,
                    character_res,
                    general_res,
                ]
            )

    submit.click(
        predict(args),
        inputs=[
            image,
            model_repo,
            custom_model,
            use_custom_model,
            general_thresh,
            general_mcut_enabled,
            character_thresh,
            character_mcut_enabled,
            replace_underscores,
            trim_general_tag_dupes,
            escape_brackets,
        ],
        outputs=[sorted_general_strings, rating, general_res, character_res],
    )
