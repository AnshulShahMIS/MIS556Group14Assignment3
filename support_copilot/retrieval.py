import hashlib
import json
import os
import threading
from dataclasses import replace
from pathlib import Path

from .config import APP_ROOT, BRANDS, MODEL_ID, MODEL_REVISION, Settings


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


class Retriever:
    def __init__(self, settings: Settings):
        import chromadb
        # Shared embedding caches must never retain a user's model credential.
        self.settings = replace(settings,api_key='')
        self.settings.runtime.mkdir(parents=True,exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(settings.runtime / 'chroma'))
        self.model = None
        self.encode_lock = threading.Lock()
        self.cases = None
        self.policies = None

    def encode(self, texts):
        with self.encode_lock:
            if self.model is None:
                os.environ.setdefault('HF_HOME',str(self.settings.runtime / 'model_cache'))
                from sentence_transformers import SentenceTransformer
                import torch
                torch.set_num_threads(min(4,os.cpu_count() or 1))
                self.model = SentenceTransformer(MODEL_ID,revision=MODEL_REVISION,device='cpu',
                                                 cache_folder=str(self.settings.runtime / 'model_cache'),trust_remote_code=False)
            return self.model.encode(texts,batch_size=64,normalize_embeddings=True,show_progress_bar=False).tolist()

    def build(self, progress=None):
        corpus = self.settings.data_dir / 'index_pairs.jsonl'
        policy_path = APP_ROOT / 'policies' / 'verified_notes.json'
        digest = hashlib.sha256(corpus.read_bytes()).hexdigest()
        policy_digest = hashlib.sha256(policy_path.read_bytes()).hexdigest()
        case_name = 'support_cases_' + digest[:12]
        policy_name = 'support_policies_' + policy_digest[:12]
        metadata = {'model':MODEL_ID,'revision':MODEL_REVISION,'corpus_sha256':digest}
        self.cases = self.client.get_or_create_collection(case_name,metadata=metadata,
                                                         configuration={'hnsw':{'space':'cosine'}})
        if self.cases.metadata.get('revision') != MODEL_REVISION:
            raise ValueError('Index embedding revision differs from the configured model. Rebuild the index.')
        rows = read_jsonl(corpus)
        vectors = None
        vector_path = self.settings.data_dir / 'index_embeddings.npz'
        if self.cases.count() != len(rows) and vector_path.exists():
            import numpy as np
            with np.load(vector_path,allow_pickle=False) as archive:
                # Git can convert record separators between CRLF and LF.
                # Accept either encoding of the exact same JSONL content.
                lf_corpus = corpus.read_bytes().replace(b'\r\n',b'\n')
                corpus_digests = {digest,hashlib.sha256(lf_corpus).hexdigest(),
                                  hashlib.sha256(lf_corpus.replace(b'\n',b'\r\n')).hexdigest()}
                if (str(archive['corpus_sha256']) not in corpus_digests or
                    str(archive['model']) != MODEL_ID or str(archive['revision']) != MODEL_REVISION or
                    archive['ids'].tolist() != [r['case_id'] for r in rows]):
                    raise ValueError('Precomputed embeddings do not match the frozen corpus/model.')
                vectors = archive['embeddings'].copy()
            if vectors.shape != (len(rows),384) or not np.isfinite(vectors).all():
                raise ValueError('Invalid precomputed embedding matrix.')
        # Only this explicit corpus is indexed. Never scan the data directory.
        if self.cases.count() != len(rows):
            for offset in range(0,len(rows),500):
                batch = rows[offset:offset+500]
                documents = [r['customer_text'] for r in batch]
                self.cases.upsert(ids=[r['case_id'] for r in batch],documents=documents,
                    embeddings=vectors[offset:offset+len(batch)].tolist() if vectors is not None else self.encode(documents),metadatas=[{
                        'brand':r['brand'],'historical_reply':r['historical_reply'],
                        'conversation_id':r['conversation_id'],'created_at':r['created_at'],
                        'source':r['source'],'doc_type':'historical_case'} for r in batch])
                if progress:
                    progress(min(offset+len(batch),len(rows)),len(rows))
        policy_rows = json.loads(policy_path.read_text(encoding='utf-8'))['documents']
        self.policies = self.client.get_or_create_collection(policy_name,
            metadata={'model':MODEL_ID,'revision':MODEL_REVISION,'policy_sha256':policy_digest},
            configuration={'hnsw':{'space':'cosine'}})
        if self.policies.count() != len(policy_rows):
            documents = [p['topic'] + '\n' + p['text'] for p in policy_rows]
            self.policies.upsert(ids=[p['policy_id'] for p in policy_rows],documents=documents,
                embeddings=self.encode(documents),metadatas=[{
                    'brand':p['brand'],'source_url':p['source_url'],'source_type':p['source_type'],
                    'retrieved_at':p['retrieved_at'],'effective_from':p.get('effective_from') or 'unknown',
                    'review_status':p['review_status'],'topic':p['topic'],'text':p['text'],
                    'demo_reply':p['demo_reply'],'doc_type':'policy'} for p in policy_rows])
        manifest = {'status':'ready','cases':self.cases.count(),'policies':self.policies.count(),
                    'case_collection':case_name,'policy_collection':policy_name,'model':MODEL_ID,
                    'revision':MODEL_REVISION,'corpus_sha256':digest,'policy_sha256':policy_digest,
                    'similarity':'cosine on normalized embeddings'}
        (self.settings.runtime/'index_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        return manifest

    def connect(self):
        path = self.settings.runtime / 'index_manifest.json'
        if not path.exists():
            raise ValueError('Build the retrieval index first.')
        manifest = json.loads(path.read_text(encoding='utf-8'))
        if manifest['revision'] != MODEL_REVISION:
            raise ValueError('Rebuild the retrieval index for the current embedding revision.')
        for path, key in [(self.settings.data_dir/'index_pairs.jsonl','corpus_sha256'),
                          (APP_ROOT/'policies/verified_notes.json','policy_sha256')]:
            if hashlib.sha256(path.read_bytes()).hexdigest() != manifest[key]:
                raise ValueError('Source data changed. Rebuild the retrieval index.')
        self.cases = self.client.get_collection(manifest['case_collection'])
        self.policies = self.client.get_collection(manifest['policy_collection'])
        if self.cases.count() != manifest['cases'] or self.policies.count() != manifest['policies']:
            raise ValueError('Retrieval index is incomplete. Rebuild it.')
        return manifest

    def search_cases(self,brand,query,top_k=5):
        if brand not in BRANDS:
            raise ValueError('Unknown brand')
        results = self.cases.query(query_embeddings=self.encode([query]),n_results=top_k,
            where={'brand':brand},include=['documents','metadatas','distances'])
        rows = []
        for case_id,text,meta,distance in zip(results['ids'][0],results['documents'][0],results['metadatas'][0],results['distances'][0]):
            if meta['brand'] != brand:
                raise ValueError('Cross-brand result rejected')
            rows.append({'case_id':case_id,'customer_text':text,**meta,'similarity':round(1-distance,4)})
        return rows

    def search_policy(self,brand,query,context='current_simulation',top_k=3):
        if brand not in BRANDS:
            raise ValueError('Unknown brand')
        if context == 'historical_replay':
            # Current FAQ notes have no verified 2017 effective dates.
            return []
        results = self.policies.query(query_embeddings=self.encode([query]),n_results=top_k,
            where={'brand':brand},include=['documents','metadatas','distances'])
        return [{'policy_id':policy_id,**meta,'similarity':round(1-distance,4)}
                for policy_id,meta,distance in zip(results['ids'][0],results['metadatas'][0],results['distances'][0])
                if meta['brand']==brand and meta['review_status']=='source_verified']
