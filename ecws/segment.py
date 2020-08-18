from typing import List

from allennlp.predictors import Predictor
from allennlp.common.util import push_python_path
from allennlp.models.archival import load_archive

import os
import importlib
import pkgutil
import json

root = os.path.dirname(os.path.abspath(__file__))

def import_module_and_submodules(package_name: str) -> None:
    """
    Import all submodules under the given package.
    Primarily useful so that people using AllenNLP as a library
    can specify their own custom packages and have their custom
    classes get loaded and registered.
    """
    importlib.invalidate_caches()

    # For some reason, python doesn't always add this by default to your path, but you pretty much
    # always want it when using `--include-package`.  And if it's already there, adding it again at
    # the end won't hurt anything.
    with push_python_path(root):
        # Import at top level
        module = importlib.import_module(package_name)
        path = getattr(module, "__path__", [])
        path_string = "" if not path else path[0]

        # walk_packages only finds immediate children, so need to recurse.
        for module_finder, name, _ in pkgutil.walk_packages(path):
            # Sometimes when you import third-party libraries that are on your path,
            # `pkgutil.walk_packages` returns those too, so we need to skip them.
            if path_string and module_finder.path != path_string:
                continue
            subpackage = f"{package_name}.{name}"
            import_module_and_submodules(subpackage)


library = 'my_library'
import_module_and_submodules(library)


class Segmenter:

    def __init__(self, archive, vocab_path, cuda_devices=-1):
        overrides = {
            "dataset_reader": {
                "type": "raw_rsv_cws_bert",
                "max_length": 256,
                "vocab_path": "bert-model/transformers-BertModel",
                "token_indexers": {
                    "tokens": "text_id"
                },
            },
            "model": {
                "model_path": ""
            }
        }
        overrides['dataset_reader']['vocab_path'] = vocab_path
        overrides['model']['model_path'] = vocab_path
        overrides = json.dumps(overrides)
        archival = load_archive(archive, cuda_devices, overrides=overrides)
        self.predictor = Predictor.from_archive(archival, 'cws')

    def seg(self, sent):
        # d = self.predictor.predict_json(json.dumps({'sent': sent}, ensure_ascii=False))
        d = self.predictor.predict_json({'sent': sent})

        return d['spans']
