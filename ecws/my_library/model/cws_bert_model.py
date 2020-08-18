from typing import Dict, List, Tuple

from allennlp.models.model import Model
from allennlp.data.vocabulary import Vocabulary
from transformers.modeling_bert import BertModel, BertForPreTraining, BertConfig
from transformers.tokenization_bert import BertTokenizer
from allennlp.training.metrics import SpanBasedF1Measure
from allennlp.data.tokenizers import Token
from transformers.tokenization_bert import _is_whitespace

import torch


def generate_spans_from_tags(tokens: List[Token], raw_text: str, tags: List, white_split=True):

    def process_inner_text(inner_text: List[str]) -> List[str]:
        output = []
        is_start = True
        inner_text = inner_text[0]
        for ch in inner_text:
            if _is_whitespace(ch):
                output.append(ch)
                is_start = True
            else:
                if is_start:
                    output.append(ch)
                    is_start = False
                else:
                    output[-1] += ch
        return output

    def merge_span_to_spans(start_, span_):
        """
        :param start_: start_指的是在外部的start_，即已经merge到spans中的token的最后一位。spans[-1].idx_end+1。可能对于即将要
        merge的span_的start不一致（span_[0].idx > start_）
        :param span_:
        :return:
        """
        if len(span_) > 0:
            span_idx = span_[0].idx
            _ = align_start(start_, span_idx)
            inner_text = [raw_text[span_[0].idx:span_[-1].idx_end+1]]
            if white_split:
                inner_text = process_inner_text(inner_text)
            spans.extend(inner_text)
            inner_pos = span_[-1].idx_end+1
            span_ = []
            return inner_pos, span_
        return start_, span_

    def align_start(start_, inner_pos):
        if start_ < inner_pos:
            spans.append(raw_text[start_:inner_pos])
        return inner_pos

    assert len(tokens) == len(tags)
    spans: List[str] = []
    start = 0
    span: List[Token] = []
    for token, tag in zip(tokens, tags):
        if tag == 'B' or tag == 'S':
            start, span = merge_span_to_spans(start, span)
            span = [token]
            if tag == 'S':
                start, span = merge_span_to_spans(start, span)
        elif tag == 'E':
            # 这里有两种写法：
            # 1. 每个E都看作是一个结尾，并将之前的span都写入spans, 并终结。例如BEEE，会生成3个span，BE, E, E
            #    但是这样子应该注意的是，连续的E之间会使得E具有S的效果。例如BEEE，BE, E, E，后面两个E实际是S的效果。
            #    故还应该加入一个开始符号，强行对齐start
            # 2. 每个E看作是结尾，写入spans但是不进行操作。但是这个效果和M似乎没有区别。
            # 故采用1。
            span.append(token)
            start, span = merge_span_to_spans(start, span)
        else:
            span.append(token)
    if len(span) > 0:
        _ = merge_span_to_spans(start, span)
    return spans


@Model.register("cws_bert")
class CWSBertModel(Model):

    def __init__(self,
                 model_path,
                 vocab: Vocabulary):
        super().__init__(vocab)
        config = BertConfig.from_pretrained(model_path)
        bert_model = BertForPreTraining(config)
        self.bert = bert_model.bert
        tags = vocab.get_index_to_token_vocabulary("tags")
        num_tags = len(tags)
        self.projection = torch.nn.Linear(768, num_tags)
        self.metric = SpanBasedF1Measure(vocab, label_encoding='BMES')

    def forward(self, tokens, attention_mask, token_type_ids, length, tags=None, metadata=None) -> Dict[str, torch.Tensor]:
        """

        :param tokens:
        :param attention_mask:
        :param token_type_ids:
        :param length: TODO (batch, 1) or (batch, )? 这个没啥，最后加一个view(-1) or view(-1, 1)就行。
        :param tags:
        :param metadata:
        :return:
        """
        output_dict = dict()
        input_ids = tokens['tokens']['tokens']
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        bert_outputs = bert_outputs[0]  # (batch, sequence, hidden_size)

        # bert_outputs包括了特殊的两个符号CLS, SEP, 并且由于之前tag和attention_mask参照tokens进行了处理。
        # 所以bert_outputs, tag, attention_mask应该对这两个符号进行处理掉。
        # 在allennlp的处理中，输入到crf中第一个位置的tag(tag[0])是必定会处理的，所以这里不能输入CLS对应的tag
        # 后面位置的tag可以通过mask进行mask掉。
        # 但是在predict阶段，还需要手动移出最后一位（根据length的长度）
        bert_outputs = bert_outputs[:, 1:, :]
        logits = self.projection(bert_outputs)  # (batch_size, seq_length, num_tags)
        batch_size, seq_length, logit_dim = logits.shape
        attention_mask = attention_mask[:, 1:].bool()
        if tags is not None:
            tags = tags[:, 1:]
            tags_flat = tags.masked_fill(~attention_mask, -100).view(-1)
            tags_flat = tags_flat.view(-1)

            flat_logtis = logits.view(-1, logit_dim)

            loss = torch.nn.functional.cross_entropy(flat_logtis, tags_flat)
            output_dict['loss'] = loss
            self.metric(logits, tags, attention_mask)

        best_path = logits.max(dim=-1)[1]
        best_path = best_path.view(batch_size, seq_length)
        output_dict['best_path'] = best_path.tolist()

        output_dict['metadata'] = metadata

        output_dict['input_ids'] = input_ids  # 已经进行了切分
        output_dict['attention_mask'] = attention_mask
        if tags is not None:
            output_dict['tags'] = tags
        best_path = [path[:mask.sum()-1] for path, mask in zip(best_path, attention_mask)]
        output_dict['best_path'] = best_path
        return output_dict

    def make_output_human_readable(
        self, output_dict
    ) -> Dict[str, torch.Tensor]:
        text_predict_tags = [[self.vocab.get_token_from_index(idx.tolist(), 'tags') for idx in path]
                             for path in output_dict['best_path']]
        output_dict.update({'text_predict_tags': text_predict_tags})
        output_dict['spans'] = []
        for tags, metadata in zip(text_predict_tags, output_dict['metadata']):
            raw_text = metadata['raw_text']
            tokens = metadata['tokens'][1:metadata['length']-1]
            spans = generate_spans_from_tags(tokens, raw_text, tags)
            output_dict['spans'].append(spans)
            metadata.pop('tokens')
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self.metric.get_metric(reset=reset)
