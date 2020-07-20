from typing import List, Dict

from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.instance import Instance
from ..tokenizers.raw_rsv_pretrained_transformer_tokenizer import RawRsvPretrainedTransformerTokenizer
from allennlp.data.fields import Field, TextField, SequenceLabelField, LabelField, MetadataField
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token
import numpy

import json


@DatasetReader.register("raw_rsv_cws_bert")
class RawRsvCWSBertDatasetReader(DatasetReader):
    """
    这里使用的tokenizer是来自可以保留raw_text tokenizer的方案。
    即是RawRsvPretrainedTransformerTokenizer.
    """

    def __init__(self,
                 max_length: int,
                 vocab_path: str,
                 token_indexers: Dict[str, TokenIndexer] = None):
        super().__init__()
        self.tokenizer = RawRsvPretrainedTransformerTokenizer(model_name=vocab_path, max_length=max_length)
        self.max_length = max_length
        self.text_max_length = max_length - 2
        self.token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer(namespace=None)}

        self.pad_token = Token(
            text=self.tokenizer.tokenizer.pad_token,
            text_id=self.tokenizer.tokenizer.pad_token_id
        )
        self.unk_token = Token(
            text=self.tokenizer.tokenizer.unk_token,
            text_id=self.tokenizer.tokenizer.unk_token_id,
            idx=-1,
            idx_end=-1
        )
        self.cls_token = Token(
            text=self.tokenizer.tokenizer.cls_token,
            text_id=self.tokenizer.tokenizer.cls_token_id,
            idx=-1,
            idx_end=-1
        )
        self.sep_token = Token(
            text=self.tokenizer.tokenizer.sep_token,
            text_id=self.tokenizer.tokenizer.sep_token_id,
            idx=-1,
            idx_end=-1
        )

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
        fields = dict()

        instance = inputs[0]
        sentence = instance['sent'][:self.text_max_length]
        tags = instance.get("tags", None)
        if tags is not None:
            tags = tags[:self.text_max_length]
            # 首尾两个是为了对齐tokens，在模型中需要去除
            tags = ['S'] + tags + ['S']
        tokens = self.tokenizer.tokenize(sentence)
        tokens = [self.cls_token] + tokens + [self.sep_token]

        # 填充到最大长度
        length = len(tokens)
        num_to_pad = self.max_length - length
        tokens = tokens + num_to_pad * [self.pad_token]
        if tags is not None:
            tags = tags + ['S'] * num_to_pad
        tokens = TextField(tokens, token_indexers=self.token_indexers)

        # (max_length, ), dim 为 1
        token_type_ids = numpy.ones(tokens.sequence_length(), dtype=int).tolist()
        attention_mask = numpy.zeros(tokens.sequence_length(), dtype=int)
        attention_mask[:length] = 1
        attention_mask = attention_mask.tolist()

        token_type_ids = SequenceLabelField(token_type_ids, tokens)
        attention_mask = SequenceLabelField(attention_mask, tokens)
        length_field = LabelField(length, skip_indexing=True)

        if tags is not None:
            tag_field = SequenceLabelField(tags, tokens, label_namespace="tags")
            fields['tags'] = tag_field

        metadata = MetadataField({
            "raw_text": sentence,
            "tokens": tokens,
            "length": length
        })
        fields.update({
            "tokens": tokens,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "length": length_field,
            "metadata": metadata
        })
        instance = Instance(fields)
        return instance


