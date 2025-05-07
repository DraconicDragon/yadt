import huggingface_hub
from PIL import Image

MODEL_REPO_PREFIX = "Camais03/"

CAMIE_MODEL_FULL = "Camais03/camie-tagger"
CAMIE_MODEL_INITIAL_ONLY = "Camais03/camie-tagger (low vram/initial only)"


class Predictor:
    def __init__(self):
        self.model = None

    def download_model(self, model_repo, full_model: bool):
        import os

        # Check if model_repo is a local directory
        if os.path.isdir(model_repo):
            # Construct paths to local model files
            metadata_path = os.path.join(model_repo, "model/metadata.json")

            if full_model:
                model_info_path = os.path.join(model_repo, "model/model_info_refined.json")
                state_dict_path = os.path.join(model_repo, "model/model_refined.pt")
            else:
                model_info_path = os.path.join(model_repo, "model/model_info_initial.json")
                state_dict_path = os.path.join(model_repo, "model/model_initial_only.pt")

            # Check if files exist
            if not os.path.exists(metadata_path):
                raise FileNotFoundError(f"Could not find metadata.json in {model_repo}/model/")
            if not os.path.exists(model_info_path):
                raise FileNotFoundError(f"Could not find model info file in {model_repo}/model/")
            if not os.path.exists(state_dict_path):
                raise FileNotFoundError(f"Could not find model state dict in {model_repo}/model/")

            return metadata_path, model_info_path, state_dict_path
        else:
            # Download from HuggingFace
            metadata_path = huggingface_hub.hf_hub_download(
                CAMIE_MODEL_FULL,
                "model/metadata.json",
            )

            if full_model:
                model_info_path = huggingface_hub.hf_hub_download(
                    CAMIE_MODEL_FULL,
                    "model/model_info_refined.json",
                )

                state_dict_path = huggingface_hub.hf_hub_download(
                    CAMIE_MODEL_FULL,
                    "model/model_refined.pt",
                )
            else:
                model_info_path = huggingface_hub.hf_hub_download(
                    CAMIE_MODEL_FULL,
                    "model/model_info_initial.json",
                )

                state_dict_path = huggingface_hub.hf_hub_download(
                    CAMIE_MODEL_FULL,
                    "model/model_initial_only.pt",
                )

            return metadata_path, model_info_path, state_dict_path

    def load_model(self, model_repo: str, **kwargs):
        import os

        # Check if it's a local path or a known model name
        if os.path.isdir(model_repo):
            # For local path, determine model type
            full_model = kwargs.pop("full_model", True)  # Default to full model for local paths
            metadata_path, model_info_path, state_dict_path = self.download_model(model_repo, full_model)
        else:
            # For known model repos
            full_model = model_repo == CAMIE_MODEL_FULL
            metadata_path, model_info_path, state_dict_path = self.download_model(CAMIE_MODEL_FULL, full_model)

        device = kwargs.pop("device", "cpu")

        from yadt.tagger_camie_model import load_model

        self.model, _, _ = load_model(
            ".",
            full=full_model,
            metadata_path=metadata_path,
            model_info_path=model_info_path,
            state_dict_path=state_dict_path,
            device=device,
        )

    def predict(self, image: Image):
        assert self.model is not None, "No model loaded"

        results = self.model.predict(image)
        tags = self.model.get_tags_from_predictions(
            results["predictions"], probabilities=results["refined_probabilities"]
        )

        return dict(tags["rating"]), dict(tags["general"]), dict(tags["character"])
