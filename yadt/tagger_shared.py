from typing import Tuple, Dict

from PIL import Image

from yadt import tagger_camie
from yadt import tagger_smilingwolf
from yadt import tagger_florence2_promptgen


class Predictor:
    def __init__(self):
        self.last_loaded_repo = None
        self.model: "Predictor" = None

    def load_model(self, model_repo: str, is_custom_model: bool, **kwargs):
        if self.last_loaded_repo == model_repo:
            return
        errors = []
        print(f"Loading model: {model_repo}")

        if model_repo.startswith(tagger_smilingwolf.MODEL_REPO_PREFIX):
            from yadt.tagger_smilingwolf import Predictor

            self.model = Predictor()
            self.model.load_model(model_repo, **kwargs)
        elif model_repo.startswith(tagger_camie.MODEL_REPO_PREFIX):
            from yadt.tagger_camie import Predictor

            self.model = Predictor()
            self.model.load_model(model_repo, **kwargs)
        elif model_repo.startswith(tagger_florence2_promptgen.MODEL_REPO_PREFIX):
            from yadt.tagger_florence2_promptgen import Predictor

            self.model = Predictor()
            self.model.load_model(model_repo, **kwargs)
        elif is_custom_model:  # Local path support
            for module_path in ["yadt.tagger_smilingwolf", "yadt.tagger_camie", "yadt.tagger_florence2_promptgen"]:
                try:
                    module = __import__(module_path, fromlist=["Predictor"])
                    self.model = module.Predictor()
                    self.model.load_model(model_repo, **kwargs)
                    break
                except Exception as e:
                    errors.append(f"{module_path}: {str(e)}")
            else:
                raise AssertionError(f"Custom model is not supported: {model_repo}\nErrors: " + "\n".join(errors))
        else:
            raise AssertionError("Model is not supported: " + model_repo)

        self.last_loaded_repo = model_repo

    def predict(self, image: Image) -> Tuple[str, Dict[str, float], Dict[str, float], Dict[str, float]]:
        assert self.model is not None, "No model loaded"
        return self.model.predict(image)


predictor = Predictor()

default_repo = tagger_smilingwolf.EVA02_LARGE_MODEL_DSV3_REPO

dropdown_list = [
    tagger_smilingwolf.SWINV2_MODEL_DSV3_REPO,
    tagger_smilingwolf.CONV_MODEL_DSV3_REPO,
    tagger_smilingwolf.VIT_MODEL_DSV3_REPO,
    tagger_smilingwolf.VIT_LARGE_MODEL_DSV3_REPO,
    tagger_smilingwolf.EVA02_LARGE_MODEL_DSV3_REPO,
    tagger_smilingwolf.MOAT_MODEL_DSV2_REPO,
    tagger_smilingwolf.SWIN_MODEL_DSV2_REPO,
    tagger_smilingwolf.CONV_MODEL_DSV2_REPO,
    tagger_smilingwolf.CONV2_MODEL_DSV2_REPO,
    tagger_smilingwolf.VIT_MODEL_DSV2_REPO,
    tagger_florence2_promptgen.FLORENCE2_PROMPTGEN_LARGE,
    tagger_florence2_promptgen.FLORENCE2_PROMPTGEN_BASE,
    tagger_camie.CAMIE_MODEL_FULL,
    tagger_camie.CAMIE_MODEL_INITIAL_ONLY,
]
