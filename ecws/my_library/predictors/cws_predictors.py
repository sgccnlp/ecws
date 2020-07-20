from typing import Dict, List, Type

from allennlp.predictors.predictor import Predictor
from allennlp.models import Archive
from allennlp.data.dataset_readers import DatasetReader

import json


@Predictor.register('cws')
class CWSPredictor(Predictor):

    def dump_line(self, outputs) -> str:
        return json.dumps(outputs, ensure_ascii=False, indent=4)

    def _json_to_instance(self, json_dict):
        return self._dataset_reader.text_to_instance(json_dict)



