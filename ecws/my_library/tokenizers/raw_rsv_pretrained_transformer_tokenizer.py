from typing import Optional, Dict, Any, Union, List

from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from .utils.raw_rsv_bert_tokenizer import RawRsvBertTokenizer
from .utils.raw_rsv_simple_token import RawRsvSimpleToken


class RawRsvPretrainedTransformerTokenizer(PretrainedTransformerTokenizer):

    def __init__(
            self,
            model_name: str,
            add_special_tokens: bool = False,
            max_length: Optional[int] = None,
            stride: int = 0,
            truncation_strategy: str = "longest_first",
            tokenizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            model_name,
            add_special_tokens,
            max_length,
            stride,
            truncation_strategy,
            tokenizer_kwargs
        )
        if tokenizer_kwargs is None:
            tokenizer_kwargs = {}
        else:
            tokenizer_kwargs = tokenizer_kwargs.copy()
        self.tokenizer = RawRsvBertTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        if add_special_tokens:
            raise NotImplementedError("目前RawRsvPretrainedTransformerTokenizer不支持add_special_tokens")

    def tokenize(self, sentence_1: Union[str, List[str]], sentence_2: str = None, start=0):

        return self.tokenizer.encode_plus(sentence_1)



