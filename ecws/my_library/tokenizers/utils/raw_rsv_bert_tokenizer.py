from typing import List, Optional, Union

from transformers.tokenization_bert import BasicTokenizer, BertTokenizer, _is_control, _is_whitespace, _is_punctuation, load_vocab

# 因为用了allennlp1_0_0所以这里的Token可以换掉了。
# from .token_1_0_0 import Token
from allennlp.data.tokenizers import Token
from .raw_rsv_simple_token import RawRsvSimpleToken

from overrides import overrides
import unicodedata
import re
import itertools
import os
import collections


def strip_whitespace(tokens: List[RawRsvSimpleToken]) -> List[RawRsvSimpleToken]:
    output = []
    is_start = True
    for token in tokens:
        for char, pos_idx, raw_char in token:
            if _is_whitespace(char):
                is_start = True
            else:
                if is_start:
                    output.append(RawRsvSimpleToken())
                    is_start = False
                output[-1].text += char
                output[-1].pos_ids.append(pos_idx)
                output[-1].raw_text += raw_char
    return output


class RawRsvBasicTokenizer(BasicTokenizer):

    def tokenize(self, token: RawRsvSimpleToken, never_split=None) -> List[RawRsvSimpleToken]:
        never_split = self.never_split + (never_split if never_split is not None else [])
        tokens = self._clean_token(token)
        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        if self.tokenize_chinese_chars:
            tokens = self._tokenize_chinese_chars(tokens)
        output_tokens = []
        for token in tokens:
            if self.do_lower_case and token not in never_split:
                token.text = token.text.lower()
                token = self._run_strip_accents(token)
            output_tokens.extend(self._run_split_on_punc(token, never_split))

        return output_tokens

    @overrides
    def _run_split_on_punc(self, token: RawRsvSimpleToken, never_split=None) -> List[RawRsvSimpleToken]:
        output = []
        is_start = True
        if token.text in never_split:
            return [token]
        for char, pos_idx, raw_char in token:
            if _is_punctuation(char):
                output.append(RawRsvSimpleToken(text=char, pos_ids=[pos_idx], raw_text=raw_char))
                is_start = True
            else:
                if is_start:
                    output.append(RawRsvSimpleToken())
                    is_start = False
                output[-1].text += char
                output[-1].pos_ids.append(pos_idx)
                output[-1].raw_text += raw_char
        return output

    @overrides
    def _run_strip_accents(self, token: RawRsvSimpleToken):
        token.text = unicodedata.normalize("NFD", token.text)
        new_token = RawRsvSimpleToken()
        for char, pos_idx, raw_char in token:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            new_token.text += char
            new_token.pos_ids.append(pos_idx)
            new_token.raw_text += raw_char
        return new_token

    def _tokenize_chinese_chars(self, tokens: List[RawRsvSimpleToken]) -> List[RawRsvSimpleToken]:
        """Adds whitespace around any CJK character."""
        output = []
        new_token = RawRsvSimpleToken()
        for token in tokens:
            for char, pos_idx, raw_char in token:
                cp = ord(char)
                if self._is_chinese_char(cp):
                    if not new_token.empty:
                        output.append(new_token)
                    new_token = RawRsvSimpleToken()
                    output.append(RawRsvSimpleToken(text=char, pos_ids=[pos_idx], raw_text=raw_char))
                else:
                    new_token.text += char
                    new_token.pos_ids.append(pos_idx)
                    new_token.raw_text += raw_char
            if not new_token.empty:
                output.append(new_token)
        return output

    def _clean_token(self, token: RawRsvSimpleToken) -> List[RawRsvSimpleToken]:
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        is_start = True
        for char, idx, raw_char in token:
            cp = ord(char)
            if cp == 0 or cp == 0xFFFD or _is_control(char) or _is_whitespace(char):
                output.append(RawRsvSimpleToken(text=" ", pos_ids=[idx], raw_text=char))
                is_start = True
            else:
                if is_start:
                    output.append(RawRsvSimpleToken())
                    is_start = False
                output[-1].text += char
                output[-1].pos_ids.append(idx)
                output[-1].raw_text += char
        return output


class RawRsvBertTokenizer(BertTokenizer):

    def __init__(
            self,
            vocab_file,
            do_lower_case=True,
            do_basic_tokenize=True,
            never_split=None,
            unk_token="[UNK]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]",
            tokenize_chinese_chars=True,
            **kwargs
    ):
        super().__init__(
            vocab_file,
            do_lower_case,
            do_basic_tokenize,
            never_split,
            unk_token,
            sep_token,
            pad_token,
            cls_token,
            mask_token,
            tokenize_chinese_chars,
            **kwargs
        )
        self.max_len_single_sentence = self.max_len - 2  # take into account special tokens
        self.max_len_sentences_pair = self.max_len - 3  # take into account special tokens

        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained "
                "model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`".format(vocab_file)
            )
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        self.do_basic_tokenize = do_basic_tokenize
        if do_basic_tokenize:
            self.basic_tokenizer = RawRsvBasicTokenizer(
                do_lower_case=do_lower_case, never_split=never_split, tokenize_chinese_chars=tokenize_chinese_chars
            )
        self.wordpiece_tokenizer = RawRsvWordpieceTokenizer(vocab=self.vocab, unk_token=self.unk_token)

    def convert_tokens(self, text: List[str]) -> List[Token]:

        all_special_tokens = self.all_special_tokens

        def lowercase_text(t):
            # convert non-special tokens to lowercase
            escaped_special_toks = [re.escape(s_tok) for s_tok in all_special_tokens]
            pattern = r"(" + r"|".join(escaped_special_toks) + r")|" + r"(.+?)"
            return re.sub(pattern, lambda m: m.groups()[0] or m.groups()[1].lower(), t)

        output = []
        start = 0
        for token in text:
            token_text = token
            if self.init_kwargs.get("do_lower_case", False):
                token_text = lowercase_text(token_text)
            token_text = token_text if token_text in self.vocab else self.unk_token
            text_id = self.vocab.get(token_text, self.unk_token_id)
            idx = start
            idx_end = start + len(token_text) - 1
            # TODO
            output.append(Token(text=token_text,
                                idx=idx,
                                idx_end=idx_end,
                                text_id=text_id))
        return output

    @overrides
    def encode_plus(
        self,
        tokens: Union[List[str], str],
        tokens_pair: Optional[Union[List[str], str]] = None,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        stride: int = 0,
        truncation_strategy: str = "longest_first",
        pad_to_max_length: bool = False,
        return_tensors: Optional[str] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        **kwargs
    ):
        """
        这个函数直接返回List[Token]
        主要原因在于，现在的*BertTokenizer的处理过程中已经引入了Token类。在进行模块切分会变得很麻烦。

        :param tokens:
        :param tokens_pair:
        :param add_special_tokens:
        :param max_length:
        :param stride:
        :param truncation_strategy:
        :param pad_to_max_length:
        :param return_tensors:
        :param return_token_type_ids:
        :param return_attention_mask:
        :param return_overflowing_tokens:
        :param return_special_tokens_mask:
        :param return_offsets_mapping:
        :param kwargs:
        :return:
        """
        if tokens_pair is not None:
            raise NotImplementedError("目前只支持一个句子的tokenization")

        def get_input_ids(text: Union[List[str], str]) -> List[Token]:

            if isinstance(text, str):
                return self.tokenize([RawRsvSimpleToken(text=text,
                                                        raw_text=text,
                                                        pos_ids=list(range(len(text))))])
            elif isinstance(text, list) and isinstance(text[0], str):
                return self.convert_tokens(text)

            else:
                raise ValueError("输入句子只能是str或者List[str]。")

        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers."
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast."
                "More information on available tokenizers at "
                "https://github.com/huggingface/transformers/pull/2674"
            )

        # Throw an error if we can pad because there is no padding token
        if pad_to_max_length and self.pad_token_id is None:
            raise ValueError(
                "Unable to set proper padding strategy as the tokenizer does not have a padding token. In this case please set the `pad_token` `(tokenizer.pad_token = tokenizer.eos_token e.g.)` or add a new pad token via the function add_special_tokens if you want to use a padding strategy"
            )

        first_tokens = get_input_ids(tokens)

        return first_tokens

    @overrides
    def _tokenize(self, token: RawRsvSimpleToken) -> List[Token]:
        split_tokens = []
        if self.do_basic_tokenize:
            for token in self.basic_tokenizer.tokenize(token, never_split=self.all_special_tokens):
                for sub_token in self.wordpiece_tokenizer.tokenize(token):
                    split_tokens.append(sub_token)
        else:
            split_tokens = self.wordpiece_tokenizer.tokenize(token)
        return split_tokens


    @overrides
    def tokenize(self, tokens: List[RawRsvSimpleToken]) -> List[Token]:
        all_special_tokens = self.all_special_tokens

        def lowercase_text(t):
            # convert non-special tokens to lowercase
            escaped_special_toks = [re.escape(s_tok) for s_tok in all_special_tokens]
            pattern = r"(" + r"|".join(escaped_special_toks) + r")|" + r"(.+?)"
            return re.sub(pattern, lambda m: m.groups()[0] or m.groups()[1].lower(), t)

        def lowercase_tokens(toks) -> List[RawRsvSimpleToken]:
            output_tokens = []
            for token in toks:
                token.text = lowercase_text(token.text)
                output_tokens.append(token)
            return output_tokens

        if self.init_kwargs.get("do_lower_case", False):
            tokens = lowercase_tokens(tokens)

        def split_token_on_special_tok(tok, token_to_split) -> List[RawRsvSimpleToken]:
            """
            这个函数中是否需要对每个sub_text进行rstrip

            :param tok:
            :param token_to_split:
            :return:
            """
            text = token_to_split.text
            pos_ids = token_to_split.pos_ids
            raw_text = token_to_split.raw_text

            tok_len = len(tok)
            result = []
            split_token_text = text.split(tok)
            start = 0
            for i, sub_text in enumerate(split_token_text):
                sub_text_len = len(sub_text)
                if i == 0 and not sub_text:
                    result += [RawRsvSimpleToken(text=tok, pos_ids=[-1]*tok_len, raw_text=tok)]
                    start += sub_text_len + tok_len
                elif i == len(split_token_text) - 1:
                    if sub_text:
                        result += [RawRsvSimpleToken(text=text[start:sub_text_len],
                                                     pos_ids=pos_ids[start:sub_text_len],
                                                     raw_text=raw_text[start:sub_text_len])]
                        start += sub_text_len
                else:
                    if sub_text:
                        result += [RawRsvSimpleToken(text=text[start:sub_text_len],
                                                     pos_ids=pos_ids[start:sub_text_len],
                                                     raw_text=raw_text[start:sub_text_len])]
                        start += sub_text_len
                    result += [RawRsvSimpleToken(text=tok, pos_ids=[-1] * tok_len, raw_text=tok)]
                    start += tok_len
            return result

        def split_tokens_on_special_toks(tok_list, tokens_to_split):
            # 仿照原来的split_on_tokens，对whitespace进行过滤。
            # 这一步是否有必要呢？如果是在保留whitespace的情况下？
            # 事实是的确需要，因为Bert的vocab中不存在whitespace，所有的whitespace会被转换为unk

            tokens_to_split = strip_whitespace(tokens_to_split)

            if not tok_list:
                return self._tokenize(tokens_to_split)

            output_tokens = []
            for tok in tok_list:
                output_tokens = []
                for token_to_split in tokens_to_split:
                    if token_to_split.text not in self.unique_added_tokens_encoder:
                        output_tokens += split_token_on_special_tok(tok, token_to_split)
                    else:
                        output_tokens += [token_to_split]
                tokens_to_split = output_tokens
            return list(
                itertools.chain.from_iterable(
                    (
                        self._tokenize(token) if token.text not in self.unique_added_tokens_encoder else [token]
                        for token in output_tokens
                    )
                )
            )

        added_tokens = self.unique_added_tokens_encoder
        tokenized_text = split_tokens_on_special_toks(added_tokens, tokens)
        return tokenized_text

    @overrides
    def prepare_for_model(
        self,
        ids: List[int],
        pair_ids: Optional[List[int]] = None,
        max_length: Optional[int] = None,
        add_special_tokens: bool = True,
        stride: int = 0,
        truncation_strategy: str = "longest_first",
        pad_to_max_length: bool = False,
        return_tensors: Optional[str] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
    ):
        """
        这里需要加入切分后每个token在原文中对应的start和end

        被PretrainedTransformerTokenizer._tokenize调用

        :param ids:
        :param pair_ids:
        :param max_length:
        :param add_special_tokens:
        :param stride:
        :param truncation_strategy:
        :param pad_to_max_length:
        :param return_tensors:
        :param return_token_type_ids:
        :param return_attention_mask:
        :param return_overflowing_tokens:
        :param return_special_tokens_mask:
        :return:
        """
        pass


class RawRsvWordpieceTokenizer:

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.unk_token_id = self.vocab[unk_token]
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, token: RawRsvSimpleToken) -> List[Token]:
        """Tokenizes a piece of text into its word pieces.

        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.

        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]

        Returns:
          A list of wordpiece tokens.
        """

        output_tokens = []
        if len(token.text) > self.max_input_chars_per_word:
            return [Token(
                text=self.unk_token,
                idx=token.pos_ids[0],
                idx_end=token.pos_ids[-1],
                text_id=self.vocab.get(self.unk_token)
            )]

        is_bad = False
        start = 0
        sub_tokens = []
        while start < len(token):
            end = len(token)
            cur_substr = None
            while start < end:
                substr = token.text[start:end]
                if start > 0:
                    substr = "##" + substr
                if substr in self.vocab:
                    cur_substr = substr
                    break
                end -= 1
            if cur_substr is None:
                is_bad = True
                break
            sub_tokens.append(Token(
                text=cur_substr,
                idx=token.pos_ids[start],
                idx_end=token.pos_ids[end-1],
                text_id=self.vocab.get(cur_substr, self.unk_token_id)
            ))
            start = end

        if is_bad:
            output_tokens.append(Token(
                text=self.unk_token,
                idx=token.pos_ids[0],
                idx_end=token.pos_ids[-1],
                text_id=self.vocab.get(self.unk_token, self.unk_token_id)
            ))
        else:
            output_tokens.extend(sub_tokens)
        return output_tokens
