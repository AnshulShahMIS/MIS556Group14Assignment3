"""Export the frozen case vectors for faster, portable cloud startup. No keys."""
import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from support_copilot.config import MODEL_ID, MODEL_REVISION, Settings
from support_copilot.retrieval import Retriever, read_jsonl

settings=Settings()
retriever=Retriever(settings)
retriever.connect()
corpus=settings.data_dir/'index_pairs.jsonl'
ids=[r['case_id'] for r in read_jsonl(corpus)]
vectors=[]
for start in range(0,len(ids),500):
    batch=ids[start:start+500]
    result=retriever.cases.get(ids=batch,include=['embeddings'])
    by_id=dict(zip(result['ids'],result['embeddings']))
    vectors.extend(by_id[item] for item in batch)
target=settings.data_dir/'index_embeddings.npz'
np.savez_compressed(target,ids=np.asarray(ids),embeddings=np.asarray(vectors,dtype=np.float32),
                    corpus_sha256=np.asarray(hashlib.sha256(corpus.read_bytes()).hexdigest()),
                    model=np.asarray(MODEL_ID),revision=np.asarray(MODEL_REVISION))
print(f'Exported {len(ids):,} vectors; {target.stat().st_size:,} bytes')
