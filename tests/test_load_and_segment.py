import pytest
import os

from ecws.segment import Segmenter


class TestLoadAndSegment:

    def setup(self):
        os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__) + '..')))
        print("#######", os.getcwd())


    def test_load_and_segment(self):
        config_path = "../archival/bert-model/transformers-BertForPreTraining"
        archival_path = "../archival/model.v3.tar.gz"

        segmenter = Segmenter(archival_path, config_path)

        sent = '我爱北京天安门'
        a = segmenter.seg(sent)
        assert a == ['我爱', '北京', '天安门']