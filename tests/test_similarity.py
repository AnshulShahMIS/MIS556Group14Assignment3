import json
import sys
from types import SimpleNamespace

import torch

from support_copilot.evaluation import SYSTEMS,add_bertscore


def test_bertscore_nested_return_contract_and_saved_results(tmp_path,monkeypatch):
    def score(candidates,references,**kwargs):
        assert len(candidates)==len(references)==3
        assert kwargs['return_hash']
        return ((torch.tensor([.8,.7,.6]),torch.tensor([.7,.6,.5]),torch.tensor([.75,.65,.55])),'test-version')
    monkeypatch.setitem(sys.modules,'bert_score',SimpleNamespace(score=score))
    (tmp_path/'cases').mkdir()
    record={'status':'complete','historical_reference':'Please check tracking.',
            'outputs':{s:{'final_reply':'Please check tracking.'} for s in SYSTEMS}}
    (tmp_path/'cases/a.json').write_text(json.dumps(record),encoding='utf-8')
    result=add_bertscore(tmp_path)
    assert result['hash']=='test-version' and result['n']==1
    assert .74<result['systems']['baseline']['f1']<.76
    assert (tmp_path/'bertscore.json').exists()
