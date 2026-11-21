from dataclasses import replace

import pytest

from support_copilot.config import Settings
from support_copilot.retrieval import Retriever


def test_chroma_metadata_filter_keeps_brands_separate(tmp_path):
    retrieval = Retriever(replace(Settings(),runtime=tmp_path))
    retrieval.cases = retrieval.client.create_collection('test_cases',configuration={'hnsw':{'space':'cosine'}})
    retrieval.cases.upsert(ids=['amazon','apple'],documents=['package late','account locked'],
        embeddings=[[1.0,0.0],[1.0,0.0]],metadatas=[
            {'brand':'AmazonHelp','historical_reply':'Track it'},
            {'brand':'AppleSupport','historical_reply':'Recover it'}])
    retrieval.encode = lambda texts:[[1.0,0.0] for _ in texts]
    result = retrieval.search_cases('AmazonHelp','account locked',1)
    assert [r['case_id'] for r in result] == ['amazon']
    with pytest.raises(ValueError):
        retrieval.search_cases('UnknownBrand','help')


def test_historical_policy_lookup_never_queries_current_store(tmp_path):
    retrieval = Retriever(replace(Settings(),runtime=tmp_path))
    retrieval.policies = None
    assert retrieval.search_policy('Delta','bag missing','historical_replay') == []


def test_precomputed_vectors_import_and_reject_wrong_source(tmp_path):
    import hashlib
    import json
    import numpy as np
    from support_copilot.config import MODEL_ID, MODEL_REVISION
    data=tmp_path/'data'
    data.mkdir()
    row={'case_id':'case1','customer_text':'late parcel','brand':'AmazonHelp',
         'historical_reply':'Please check tracking.','conversation_id':'thread1',
         'created_at':'2017','source':'TWCS'}
    corpus=data/'index_pairs.jsonl'
    corpus.write_text(json.dumps(row)+'\n',encoding='utf-8')
    archive=dict(ids=np.asarray(['case1']),embeddings=np.ones((1,384),dtype=np.float32),
                 corpus_sha256=np.asarray(hashlib.sha256(corpus.read_bytes()).hexdigest()),
                 model=np.asarray(MODEL_ID),revision=np.asarray(MODEL_REVISION))
    np.savez_compressed(data/'index_embeddings.npz',**archive)
    retrieval=Retriever(replace(Settings(),runtime=tmp_path/'runtime',data_dir=data))
    def policies_only(texts):
        assert 'late parcel' not in texts,'Case vectors must come from the archive'
        return [[1.0]*384 for _ in texts]
    retrieval.encode=policies_only
    assert retrieval.build()['cases']==1
    assert retrieval.connect()['cases']==1
    archive['corpus_sha256']=np.asarray('wrong')
    np.savez_compressed(data/'index_embeddings.npz',**archive)
    other=Retriever(replace(Settings(),runtime=tmp_path/'other',data_dir=data))
    with pytest.raises(ValueError,match='frozen corpus/model'):
        other.build()
