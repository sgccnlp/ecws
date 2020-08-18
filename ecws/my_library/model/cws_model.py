from typing import Dict

from allennlp.models.model import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.conditional_random_field import ConditionalRandomField, allowed_transitions
from transformers.modeling_bert import BertModel, BertForPreTraining, BertConfig
from transformers.tokenization_bert import BertTokenizer

import torch


@Model.register("cws_model")
class CWSModel(Model):

    def __init__(self,
                 model_path,
                 vocab: Vocabulary):
        super().__init__(vocab)
        self.pretrained_tokenizer = BertForPreTraining.from_pretrained(model_path)
        config = BertConfig.from_pretrained(model_path)
        bert_model = BertForPreTraining(config)
        self.bert = bert_model.bert
        tags = vocab.get_index_to_token_vocabulary("tags")
        num_tags = len(tags)
        constraints = allowed_transitions(constraint_type="BMES", labels=tags)
        self.projection = torch.nn.Linear(768, num_tags)
        self.crf = ConditionalRandomField(num_tags=num_tags, constraints=constraints, include_start_end_transitions=False)

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
        logits = self.projection(bert_outputs)
        log_likelihood = torch.nn.functional.log_softmax(logits, -1)
        attention_mask = attention_mask[:, 1:]
        if tags is not None:
            tags = tags[:, 1:]

            loss = -self.crf(log_likelihood, tags, attention_mask)
            output_dict['loss'] = loss

        # 运行viterbi解码
        best_path = self.crf.viterbi_tags(logits, attention_mask)
        output_dict['best_path'] = best_path

        output_dict['metadata'] = metadata

        output_dict['input_ids'] = input_ids[:, 1:]  # 已经进行了切分
        output_dict['attention_mask'] = attention_mask
        if tags is not None:
            output_dict['tags'] = tags
        best_path = [path[0][:mask.sum()] for path, mask in zip(best_path, attention_mask)]
        output_dict['best_path'] = best_path
        return output_dict

    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        text_predict_tags = [[self.vocab.get_token_from_index(idx, 'tags') for idx in path]
                             for path in output_dict['best_path']]
        output_dict.update({'text_predict_tags': text_predict_tags})
        return output_dict
