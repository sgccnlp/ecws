from typing import List


class RawRsvSimpleToken:
    """
    这里只保存
    text: 处理后的text
    pos_ids: text中每个字符在原文中对应的位置
    raw_text: text中每个字符在原文中对应的字符。
    所以三者的长度都应该相等。len(text) == len(pos_ids) == len(raw_text)

    此外，为什么不存储完整的raw_text对应的raw_pos_ids，根本问题在于在保证raw_text字符和text字符一一对应的情况下，
    两者的pos_ids应该也是一一对应并且一致。但如果text和raw_text的字符不是一一对应，又回回到最开始的问题：
    当继续对text进行切分的时候，raw_text中不与text对应的字符应该如何切分。

    注意，如果将text转换为特殊符号，例如[UNK]后，raw_text和pos_ids应该还是和[UNK]对应的原始字符串保持一致。
    这就会导致len(self.text) != len(self.raw_text)。
    可能的解决方案是：不使用RawRsvSimpleToken来存储这种类型。并且这种转换应该位于预处理分词中的最后一步。
    """

    def __init__(self, text=None, pos_ids=None, raw_text=None):
        self.text: str = text or ''
        self.pos_ids: List[int] = pos_ids or []
        self.raw_text: str = raw_text or ''

    @property
    def empty(self):
        return len(self.raw_text) == 0

    def __iter__(self):
        assert len(self.text) == len(self.pos_ids) == len(self.raw_text)
        for char, pos_idx, raw_char in zip(self.text, self.pos_ids, self.raw_text):
            yield char, pos_idx, raw_char

    def __len__(self):
        return len(self.raw_text)

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            raise ValueError(f"Index only support integer, but get {type(idx)}.")
        return self.text[idx], self.pos_ids[idx], self.raw_text[idx]

    def __str__(self):
        return self.text
