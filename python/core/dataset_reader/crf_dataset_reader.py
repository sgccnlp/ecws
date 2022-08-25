from typing import List

from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.fields import Field, TextField, SequenceLabelField, LabelField, MetadataField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
import numpy

import json


@DatasetReader.register("cws_dataset_reader")
class CRFDatasetReader(DatasetReader):

    def __init__(self,
                 max_length: int):
        super().__init__()
        self.token_indexers = {'tokens': SingleIdTokenIndexer()}
        self.max_length = max_length

    def _read(self, path: str):
        with open(path, encoding='utf8') as fp:
            for line in fp:
                instance = json.loads(line)
                yield self.text_to_instance(instance)

    def text_to_instance(self, *inputs) -> Instance:
        """
        这里需要注意的是，在对输入text进行处理的时候，因为bert的原因，会加入CLS和SEP两个特殊符号，导致tag和text的序列长度不一致。
        因此需要认为地加入tag使两者长度保持一致。
        同时，因为要pad_to_max_length，这一步也需要人为操作。
        :param inputs:
        :return:
        """
        instance = inputs[0]
        sentence = instance['sent'][:self.text_max_length]
        tags = instance.get("tags", None)[:self.text_max_length]

        # 填充到最大长度
        length = len(tokens)
        num_to_pad = self.max_length - length
        tokens = tokens + num_to_pad * [self.pad_token]
        tags = tags + ['S'] * num_to_pad
        tokens = TextField(tokens, token_indexers=self.token_indexers)

        # (max_length, ), dim 为 1
        token_type_ids = numpy.ones(tokens.sequence_length(), dtype=int).tolist()
        attention_mask = numpy.zeros(tokens.sequence_length(), dtype=int)
        attention_mask[:length] = 1
        attention_mask = attention_mask.tolist()

        token_type_ids = SequenceLabelField(token_type_ids, tokens)
        attention_mask = SequenceLabelField(attention_mask, tokens)
        length = LabelField(length, skip_indexing=True)

        tag_field = SequenceLabelField(tags, tokens, label_namespace="tags")
        metadata = MetadataField({
            "raw_text": ' '.join(sentence)
        })
        instance = Instance({
            "tokens": tokens,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "length": length,
            "tags": tag_field,
            "metadata": metadata
        })
        return instance


