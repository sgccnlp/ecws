from typing import List, Tuple

from transformers.tokenization_bert import BertTokenizer, BasicTokenizer, _is_control, _is_whitespace

from overrides import overrides
import unicodedata


class RawRsvBasicTokenizer(BasicTokenizer):
    """
    和.raw_rsv_wordpiece_tokenizer.RawRsvWordpieceTokenizer联合使用，
    达到能够保留原来空格和还原加入特殊符号的分词目的。
    因为这个在allennlp.data.tokenizers.PretrainedTransformerTokenizer, BertTokenizer,
    transformers.tokenization_utils.PreTrainedTokenizer中的调用顺序是第一位。
    所以*BasicTokenizer的输入只用考虑raw_text，而不会接受其它的输入。故tokenize的输入只有text
    """
    def tokenize(self, text, never_split=None):
        never_split = self.never_split + (never_split if never_split is not None else [])
        # _clean_text本身完成了whitespace_tokenize的工作。
        tokens, raw_rsv_tokens = self._clean_text(text)
        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        if self.tokenize_chinese_chars:
            tokens, raw_rsv_tokens = self._tokenize_chinese_chars(tokens, raw_rsv_tokens)
        split_tokens = []
        split_raw_rsv_tokens = []
        for token, raw_rsv_token in zip(tokens, raw_rsv_tokens):
            if self.do_lower_case and token not in never_split:
                token = token.lower()
                # 虽然_run_strip_accents对重音符号进行删除。
                # 但是只要保证tokens中的每个token都和raw_rsv_tokens中的每个raw_rsv_token
                # 都对应上即可。即是某个token被strip成了''
                token = self._run_strip_accents(token)
            tokens_, raw_rsv_tokens_ = self._run_split_on_punc(token, raw_rsv_token, never_split)
            split_tokens.extend(tokens_)
            split_raw_rsv_tokens.extend(raw_rsv_tokens_)

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    @overrides
    def _run_split_on_punc(self, text, raw_rsv_text, never_split=None) -> Tuple[List[str], List[str]]:
        output_tokens, output_raw_rsv_tokens = [], []
        """Splits punctuation on a piece of text."""
        if never_split is not None and text in never_split:
            return [text], [raw_rsv_text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]


        return output_tokens, output_raw_rsv_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    @overrides
    def _tokenize_chinese_chars(self, tokens: List[str], raw_rsv_tokens: List[str]) -> Tuple[List[str], List[str]]:
        output_tokens = []
        output_raw_rsv_tokens = []
        output_token = ''
        output_raw_rsv_token = ''
        for token, raw_rsv_token in zip(tokens, raw_rsv_tokens):
            for char, raw_rsv_char in zip(token, raw_rsv_token):
                cp = ord(char)
                if self._is_chinese_char(cp):
                    if len(token) > 0:
                        output_tokens.append(output_token)
                        output_raw_rsv_tokens.append(output_raw_rsv_token)
                    output_tokens.append(char)
                    output_raw_rsv_tokens.append(raw_rsv_char)
                    output_token = ''
                    output_raw_rsv_token = ''
                else:
                    output_token += char
                    output_raw_rsv_token += raw_rsv_char
            if len(output_token) > 0:
                output_tokens.append(output_token)
                output_raw_rsv_tokens.append(output_raw_rsv_token)
        return output_tokens, output_raw_rsv_tokens

    @overrides
    def _clean_text(self, text) -> Tuple[List[str], List[str]]:
        """
        相较于原本的transformers.tokenization_bert.BertTokenizer._clean_text,
        将控制字符也输出为空格。主要目的是保持输出前后的长度一致。
        同时也应该输出原始text
        :param text:
        :return: (cleaned_text, raw_text, )
        """
        output = []
        raw_rsv_output = []
        token = ''
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xFFFD or _is_control(char) or _is_whitespace(char):
                if len(token) > 0:
                    output.append(token)
                    raw_rsv_output.append(token)
                token = ''
                output.append(" ")
            else:
                token += char
            if len(token) > 0:
                output.append(token)
                raw_rsv_output.append(token)
        assert sum(len(token) for token in output) == sum(len(token) for token in raw_rsv_output)
        return output, raw_rsv_output


class RawRsvBertTokenizer(BertTokenizer):
    """
    和.raw_rsv_wordpiece_tokenizer.RawRsvWordpieceTokenizer联合使用，
    达到能够保留原来空格和还原加入特殊符号的分词目的。
    """
    @overrides
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
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )
        pass

    @overrides
    def _tokenize(self, text):
        pass
