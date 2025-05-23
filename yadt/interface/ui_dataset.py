import os
import gradio as gr

from PIL import Image

from yadt import tagger_shared
from yadt import process_prediction
from yadt.interface import ui_utils
from yadt.interface.shared.model_selector import create_model_selector
from yadt.interface.shared.wd_tagger_threshold import create_threshold_options


def temp_folder_gallery_path(args, name: str):
    return f"{args.tempfolder}/{name}.jpeg"


def save_caption_for_image_path(image_path: str, caption: str, overwrite_current_caption: bool = False):
    caption_file_path = image_path[: image_path.rindex(".")] + ".txt"

    if not os.path.exists(caption_file_path) or overwrite_current_caption:
        with open(caption_file_path, "w") as f:
            f.write(caption)


def process_dataset_folder(args):
    import zlib
    import pickle
    import hashlib

    from yadt.db_dataset import db

    def hash_file(path: str):
        with open(path, "rb") as f:
            hash = hashlib.sha256(f.read())
            return hash.digest()

    def encode_results(*args):
        return zlib.compress(pickle.dumps(args))

    def decode_results(data: bytes):
        return pickle.loads(zlib.decompress(data))

    warning_default = [None, {}, [], gr.Column(visible=False), {}, {}, {}]

    @ui_utils.gradio_warning(default=warning_default)
    def _process_dataset_folder(
        folder: str,
        model_repo: str,
        general_thresh: float,
        general_mcut_enabled: bool,
        character_thresh: float,
        character_mcut_enabled: bool,
        replace_underscores: bool,
        trim_general_tag_dupes: bool,
        escape_brackets: bool,
        overwrite_current_caption: bool,
        prefix_tags: str,
        keep_tags: str,
        ban_tags: str,
        map_tags: str,
        progress=gr.Progress(),
    ):
        warning_default[0] = folder
        warning_default[2] = [folder, None]

        assert len(folder) > 0, "No folder given"
        assert os.path.isdir(folder), "Folder either doesn't exist or is not a folder"

        db.update_recent_datasets(folder)

        # predictor.load_model(model_repo)

        files = os.listdir(folder)
        files = list(
            filter(lambda f: not f.endswith(".txt") and not f.endswith(".npz") and not f.endswith(".json"), files)
        )

        all_count = 0
        all_images = []
        all_rating = dict()
        all_character_res = dict()
        all_general_res = dict()

        for index, file in progress.tqdm(list(enumerate(files)), desc=folder):
            image_path = folder + "/" + file

            file_hash = hash_file(image_path)
            file_hash_hex = file_hash.hex()

            try:
                image = Image.open(image_path)
            except Exception as e:
                continue

            cache = db.get_dataset_cache(file_hash, model_repo)
            if cache is not None:
                rating, general_res, character_res = decode_results(cache)
            else:
                tagger_shared.predictor.load_model(model_repo, device=args.device)
                rating, general_res, character_res = tagger_shared.predictor.predict(image)

                db.set_dataset_cache(file_hash, model_repo, folder, encode_results(rating, general_res, character_res))

            sorted_general_strings, rating, general_res, character_res = process_prediction.post_process_prediction(
                rating,
                general_res,
                character_res,
                general_thresh,
                general_mcut_enabled,
                character_thresh,
                character_mcut_enabled,
                general_thresh,
                False,
                character_thresh,
                False,
                replace_underscores,
                trim_general_tag_dupes,
                escape_brackets,
                prefix_tags,
                keep_tags,
                ban_tags,
                map_tags,
            )

            manual_edit = db.get_dataset_edit(folder, file_hash)
            if manual_edit is not None:
                previous_edit, new_edit = manual_edit
                sorted_general_strings_post = process_prediction.post_process_manual_edits(
                    previous_edit, new_edit, sorted_general_strings
                )
            else:
                sorted_general_strings_post = sorted_general_strings

            # print('===', file)
            # print(sorted_general_strings)
            # print('')

            all_count += 1

            temp_image_path = temp_folder_gallery_path(args, file_hash_hex)
            if not os.path.exists(temp_image_path):
                image.convert("RGB").save(temp_image_path, quality=85)

            all_images.append((file_hash_hex, [image_path, sorted_general_strings, sorted_general_strings_post]))

            for k in rating.keys():
                all_rating[k] = all_rating.get(k, 0) + rating[k]

            for k in character_res.keys():
                all_character_res[k] = all_character_res.get(k, 0) + 1

            for k in general_res.keys():
                all_general_res[k] = all_general_res.get(k, 0) + 1

            save_caption_for_image_path(
                image_path, sorted_general_strings_post, overwrite_current_caption=overwrite_current_caption
            )

        for k in all_rating.keys():
            all_rating[k] = all_rating[k] / all_count

        for k in all_character_res.keys():
            all_character_res[k] = all_character_res[k] / all_count

        for k in all_general_res.keys():
            all_general_res[k] = all_general_res[k] / all_count

        return [
            gr.Dropdown(choices=load_recent_datasets()),
            all_images,
            [folder, None],
            gr.Column(visible=True),
            all_rating,
            all_general_res,
            all_character_res,
        ]

    return _process_dataset_folder


def process_dataset_gallery(args):
    @ui_utils.gradio_warning
    def _process_dataset_gallery(all_images: list[tuple[str, tuple[str, str, str]]], filters: list[str]):
        if len(filters) == 0:
            return [(temp_folder_gallery_path(args, image), image) for image, tags in all_images]

        filters = set(filters)

        return [
            (temp_folder_gallery_path(args, image), image)
            for image, (_, _, tags) in all_images
            if set([tag.strip() for tag in tags.split(",")]).issuperset(filters)
        ]

    return _process_dataset_gallery


def process_dataset_gallery_filters(args):
    @ui_utils.gradio_warning
    def _process_dataset_gallery(all_images: list[tuple[str, tuple[str, str, str]]]):
        all_image_dict = {}

        for _, (_, _, tags) in all_images:
            for tag in tags.split(","):
                tag = tag.strip()

                if tag in all_image_dict:
                    all_image_dict[tag] += 1
                else:
                    all_image_dict[tag] = 1

        return gr.Dropdown(
            choices=[
                (f"{tag} [{count}]", tag)
                for tag, count in sorted(all_image_dict.items(), key=lambda item: item[1], reverse=True)
            ]
        )

    return _process_dataset_gallery


@ui_utils.gradio_warning(default=[])
def load_recent_datasets():
    from yadt.db_dataset import db

    return db.get_recent_datasets()


def load_dataset_settings(args):
    model_repo_default = tagger_shared.default_repo
    general_thresh_default = args.score_general_threshold
    general_mcut_enabled_default = "False"
    character_thresh_default = args.score_character_threshold
    character_mcut_enabled_default = "False"
    replace_underscores_default = "True"
    trim_general_tag_dupes_default = "True"
    escape_brackets_default = "False"
    overwrite_current_caption_default = "False"
    prefix_tags_default = ""
    keep_tags_default = ""
    ban_tags_default = ""
    map_tags_default = ""

    @ui_utils.gradio_warning(
        default=[
            model_repo_default,
            general_thresh_default,
            general_mcut_enabled_default,
            character_thresh_default,
            character_mcut_enabled_default,
            replace_underscores_default,
            trim_general_tag_dupes_default,
            escape_brackets_default,
            overwrite_current_caption_default,
            prefix_tags_default,
            keep_tags_default,
            ban_tags_default,
            map_tags_default,
        ]
    )
    def _load_dataset_settings(folder: str):
        from yadt.db_dataset import db

        model_repo = str(db.get_dataset_setting(folder, "model_repo", default=model_repo_default))
        general_thresh = float(db.get_dataset_setting(folder, "general_thresh", default=general_thresh_default))
        general_mcut_enabled = (
            db.get_dataset_setting(folder, "general_mcut_enabled", default=general_mcut_enabled_default)
        ) == "True"
        character_thresh = float(db.get_dataset_setting(folder, "character_thresh", default=character_thresh_default))
        character_mcut_enabled = (
            db.get_dataset_setting(folder, "character_mcut_enabled", default=character_mcut_enabled_default)
        ) == "True"
        replace_underscores = (
            db.get_dataset_setting(folder, "replace_underscores", default=replace_underscores_default)
        ) == "True"
        trim_general_tag_dupes = (
            db.get_dataset_setting(folder, "trim_general_tag_dupes", default=trim_general_tag_dupes_default)
        ) == "True"
        escape_brackets = (db.get_dataset_setting(folder, "escape_brackets", default=escape_brackets_default)) == ""
        overwrite_current_caption = (
            db.get_dataset_setting(folder, "overwrite_current_caption", default=overwrite_current_caption_default)
        ) == "True"
        prefix_tags = str(db.get_dataset_setting(folder, "prefix_tags", default=prefix_tags_default))
        keep_tags = str(db.get_dataset_setting(folder, "keep_tags", default=keep_tags_default))
        ban_tags = str(db.get_dataset_setting(folder, "ban_tags", default=ban_tags_default))
        map_tags = str(db.get_dataset_setting(folder, "map_tags", default=map_tags_default))

        return [
            model_repo,
            general_thresh,
            general_mcut_enabled,
            character_thresh,
            character_mcut_enabled,
            replace_underscores,
            trim_general_tag_dupes,
            escape_brackets,
            overwrite_current_caption,
            prefix_tags,
            keep_tags,
            ban_tags,
            map_tags,
        ]

    return _load_dataset_settings


def save_dataset_settings(args):
    @ui_utils.gradio_warning
    def _save_dataset_settings(
        folder: str,
        model_repo: str,
        general_thresh: float,
        general_mcut_enabled: bool,
        character_thresh: float,
        character_mcut_enabled: bool,
        replace_underscores: bool,
        trim_general_tag_dupes: bool,
        escape_brackets: bool,
        overwrite_current_caption: bool,
        prefix_tags: str,
        keep_tags: str,
        ban_tags: str,
        map_tags: str,
    ):
        from yadt.db_dataset import db

        db.set_dataset_setting(folder, "model_repo", str(model_repo))
        db.set_dataset_setting(folder, "general_thresh", str(general_thresh))
        db.set_dataset_setting(folder, "general_mcut_enabled", str(general_mcut_enabled))
        db.set_dataset_setting(folder, "character_thresh", str(character_thresh))
        db.set_dataset_setting(folder, "character_mcut_enabled", str(character_mcut_enabled))
        db.set_dataset_setting(folder, "replace_underscores", str(replace_underscores))
        db.set_dataset_setting(folder, "trim_general_tag_dupes", str(trim_general_tag_dupes))
        db.set_dataset_setting(folder, "escape_brackets", str(escape_brackets))
        db.set_dataset_setting(folder, "overwrite_current_caption", str(overwrite_current_caption))
        db.set_dataset_setting(folder, "prefix_tags", str(prefix_tags))
        db.set_dataset_setting(folder, "keep_tags", str(keep_tags))
        db.set_dataset_setting(folder, "ban_tags", str(ban_tags))
        db.set_dataset_setting(folder, "map_tags", str(map_tags))

    return _save_dataset_settings


def on_gallery_select(
    selection: tuple[str, str], all_images: list[tuple[str, tuple[str, str, str]]], event: gr.SelectData
):
    _selection = event.value["caption"]
    caption = next(filter(lambda image: image[0] == _selection, all_images), [None, [None, None, None]])[1][2]

    assert caption is not None, f"Could not find caption for the selected image: {_selection}"

    return [
        gr.Text(value=caption, interactive=True),
        gr.Column(visible=True),
        [selection[0], _selection],
    ]


def on_gallery_deselect(selection: tuple[str, str]):
    return [
        gr.Text(value=None, interactive=False),
        gr.Column(visible=False),
        [selection[0], None],
    ]


def on_gallery_reset(selection: tuple[str, str], all_images: list[tuple[str, tuple[str, str, str]]]):
    if selection[1] is None:
        return None

    selection = selection[1]
    return next(filter(lambda image: image[0] == selection, all_images), [None, [None, None, None]])[1][1]


def on_gallery_reload(selection: tuple[str, str], all_images: list[tuple[str, tuple[str, str, str]]]):
    if selection[1] is None:
        return None

    selection = selection[1]
    return next(filter(lambda image: image[0] == selection, all_images), [None, [None, None, None]])[1][2]


@ui_utils.gradio_warning
def on_gallery_save(selection: tuple[str, str], all_images: list[tuple[str, tuple[str, str, str]]], caption: str):
    from yadt.db_dataset import db

    assert len(selection) > 0, "No gallery image selected"

    folder, selection = selection
    gallery_item = next(filter(lambda image: image[0] == selection, all_images), [None, [None, None, None]])

    assert gallery_item[0] is not None, f"Could not find selected image: {selection}"

    file_hash_hex = gallery_item[0]
    file_hash = bytes.fromhex(file_hash_hex)
    image_path, initial_edit, _ = gallery_item[1]

    save_caption_for_image_path(image_path, caption, overwrite_current_caption=True)
    db.set_dataset_edit(folder, file_hash, initial_edit, caption)

    try:
        all_images_i = list(map(lambda i: i[0], all_images)).index(file_hash_hex)
        all_images[all_images_i] = [file_hash_hex, [image_path, initial_edit, caption]]
    except ValueError:
        gr.Warning(f"Could not update caption for selected image: {selection}")

    return all_images


def ui(args):
    with gr.Blocks() as page:
        with gr.Row():
            with gr.Column(variant="panel"):
                with gr.Row(equal_height=True):
                    # folder = gr.Textbox(label="Select folder:", scale=1)
                    folder = gr.Dropdown(
                        label="Select folder:",
                        choices=load_recent_datasets(),
                        allow_custom_value=True,
                        scale=1,
                    )

                    load_folder = gr.Button(value="Load", variant="primary", scale=0)

                gr.HTML(
                    '<p style="margin-top: -1em"><i>Dataset settings are saved on submit. Use the load button to reload them.</i></p>'
                )

                model_repo, custom_model, use_custom_model = create_model_selector()

                (
                    general_thresh,
                    general_mcut_enabled,
                    character_thresh,
                    character_mcut_enabled,
                ) = create_threshold_options(args, include_mcut_checkboxes=True)

                with gr.Row():
                    overwrite_current_caption = gr.Checkbox(
                        value=False,
                        label="Overwrite existing captions",
                        scale=1,
                    )
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
                        value=False,
                        label="Escape brackets (for webui)",
                        scale=1,
                    )

                with gr.Column(variant="panel"):
                    prefix_tags = gr.Textbox(label="Prefix tags:", placeholder="tag1, tag2, ...")
                    keep_tags = gr.Textbox(label="Keep tags:", placeholder="tag1, tag2, ...")
                    ban_tags = gr.Textbox(label="Ban tags:", placeholder="tag1, tag2, ...")
                    map_tags = gr.Textbox(
                        label="Map tags",
                        placeholder='one or more lines of "tag1, tag2, ... : tag"',
                        lines=5,
                        max_lines=100,
                    )

                    gr.HTML(
                        """
                        <p>Prefixing tags</p>
                        <p><i>Adding any tags to this will sort the tags and add them before a "BREAK" tag.</i></p>
                        <br>
                        <p>Mapping tags</p>
                        <p><i>You can map certain one or more tags to different tags. Examples: </i></p>
                        <p style="padding-left: 1em"><i>* BAD_TAG : GOOD_TAG</i></p>
                        <p style="padding-left: 1em"><i>* 2girl : 2girls, GIRL_ONE, GIRL_TWO</i></p>
                    """
                    )

                with gr.Row():
                    clear = gr.ClearButton(
                        components=[
                            folder,
                            model_repo,
                            general_thresh,
                            general_mcut_enabled,
                            character_thresh,
                            character_mcut_enabled,
                            replace_underscores,
                            trim_general_tag_dupes,
                            escape_brackets,
                            overwrite_current_caption,
                            prefix_tags,
                            keep_tags,
                            ban_tags,
                            map_tags,
                        ],
                        variant="secondary",
                        size="lg",
                    )

                    submit = gr.Button(value="Submit", variant="primary", size="lg")

            with gr.Column():
                with gr.Column(variant="panel"):
                    # FIXME: gr.JSON deals stateless requests, but it also sends all the captions over
                    gallery_cache = gr.JSON(visible=False)
                    gallery_selection = gr.JSON(visible=False)

                    with gr.Column(visible=False) as gallery_tags_filter:
                        gr.HTML(
                            "<h3>Dataset gallery</h3><p><i>Use the dropdown below to filter the images by tags</i></p>"
                        )
                        gallery_tags_filter_dropdown = gr.Dropdown(
                            choices=["1girl", "2girls"],
                            label="Filter by tag",
                            multiselect=True,
                            interactive=True,
                            show_label=False,
                            container=False,
                        )

                    gallery = gr.Gallery(interactive=False, columns=3)

                    with gr.Column(visible=False) as gallery_tags_view:
                        gallery_tags_edit = gr.Text(
                            interactive=False,
                            show_label=False,
                            container=False,
                            placeholder="Select an image to view the resulting tags.",
                        )

                        with gr.Row():
                            gallery_tags_reset = gr.Button(value="Reset")
                            gallery_tags_reload = gr.Button(value="Reload")
                            gallery_tags_save = gr.Button(value="Save", variant="primary")

                        gr.HTML(
                            '<p>Editing dataset tags is still <i><b>experimental</b></i>.</p><p style="font-size: 0.9em"><i><b>Reset</b> will clear any changes made previously, and set the tags back to the original model tags (using the rules set on the left side). <br> <b>Reload</b> will clear any temporary manual changes and load the latest modified tags from the local database. <br> <b>Save</b> will update the database and the caption file. Neither reset nor reload will update the database or caption file until the save button is clicked.</i></p>'
                        )

                with gr.Column(variant="panel"):
                    rating = gr.Label(label="Rating")
                    character_res = gr.Label(label="Output (characters)")
                    general_res = gr.Label(label="Output (tags)")

                    clear.add(
                        [
                            rating,
                            character_res,
                            general_res,
                        ]
                    )

    submit.click(
        process_dataset_folder(args),
        inputs=[
            folder,
            model_repo,
            general_thresh,
            general_mcut_enabled,
            character_thresh,
            character_mcut_enabled,
            replace_underscores,
            trim_general_tag_dupes,
            escape_brackets,
            overwrite_current_caption,
            prefix_tags,
            keep_tags,
            ban_tags,
            map_tags,
        ],
        outputs=[
            folder,
            gallery_cache,
            gallery_selection,
            gallery_tags_filter,
            rating,
            general_res,
            character_res,
        ],
    )

    gallery_cache.change(
        process_dataset_gallery_filters(args),
        inputs=[gallery_cache],
        outputs=[gallery_tags_filter_dropdown],
    )

    gallery_cache.change(
        process_dataset_gallery(args),
        inputs=[gallery_cache, gallery_tags_filter_dropdown],
        outputs=[gallery],
    )

    gallery_tags_filter_dropdown.change(
        process_dataset_gallery(args),
        inputs=[gallery_cache, gallery_tags_filter_dropdown],
        outputs=[gallery],
    )

    gallery.select(
        on_gallery_select,
        inputs=[gallery_selection, gallery_cache],
        outputs=[gallery_tags_edit, gallery_tags_view, gallery_selection],
    )

    gallery.preview_close(
        on_gallery_deselect,
        inputs=[gallery_selection],
        outputs=[gallery_tags_edit, gallery_tags_view, gallery_selection],
    )

    gallery_tags_reset.click(
        on_gallery_reset,
        inputs=[gallery_selection, gallery_cache],
        outputs=[gallery_tags_edit],
    )

    gallery_tags_reload.click(
        on_gallery_reload,
        inputs=[gallery_selection, gallery_cache],
        outputs=[gallery_tags_edit],
    )

    gallery_tags_save.click(
        on_gallery_save,
        inputs=[gallery_selection, gallery_cache, gallery_tags_edit],
        outputs=[gallery_cache],
    )

    dataset_settings = [
        model_repo,
        general_thresh,
        general_mcut_enabled,
        character_thresh,
        character_mcut_enabled,
        replace_underscores,
        trim_general_tag_dupes,
        escape_brackets,
        overwrite_current_caption,
        prefix_tags,
        keep_tags,
        ban_tags,
        map_tags,
    ]

    folder.select(
        load_dataset_settings(args),
        inputs=[folder],
        outputs=dataset_settings,
    )

    page.load(
        load_dataset_settings(args),
        inputs=[folder],
        outputs=dataset_settings,
    )

    load_folder.click(
        load_dataset_settings(args),
        inputs=[folder],
        outputs=dataset_settings,
    )

    load_folder.click(
        lambda: gr.Dropdown(choices=load_recent_datasets()),
        outputs=folder,
    )

    submit.click(
        save_dataset_settings(args),
        inputs=[folder] + dataset_settings,
    )
