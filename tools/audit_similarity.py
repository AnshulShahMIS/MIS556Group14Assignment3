"""Flag similar held-out inputs without changing the frozen split."""
import json
from pathlib import Path

import numpy as np

from support_copilot.config import APP_ROOT, BRANDS, Settings, MODEL_ID, MODEL_REVISION
from support_copilot.retrieval import Retriever, read_jsonl


def main():
    settings = Settings.from_env()
    retriever = Retriever(settings)
    retriever.connect()
    rows = read_jsonl(settings.data_dir/'test_pairs.jsonl')
    comparisons = []
    for brand in BRANDS:
        group = [r for r in rows if r['brand'] == brand]
        result = retriever.cases.query(query_embeddings=retriever.encode([r['customer_text'] for r in group]),
            n_results=1,where={'brand':brand},include=['documents','distances','metadatas'])
        for i,row in enumerate(group):
            comparisons.append({'case_id':row['case_id'],'brand':brand,
                'customer_text':row['customer_text'],'nearest_index_case_id':result['ids'][i][0],
                'nearest_index_text':result['documents'][i][0],
                'similarity':round(1-result['distances'][i][0],6)})
    similarities = [r['similarity'] for r in comparisons]
    report = {'method':'Approximate nearest-neighbor cosine search, same-brand, top 1 per held-out input.',
        'model':MODEL_ID,'revision':MODEL_REVISION,'threshold':0.95,
        'interpretation':'Similarity flags are candidates for manual review, not proven leakage. Split is unchanged.',
        'n':len(rows),'flagged_n':sum(s>=0.95 for s in similarities),
        'median_similarity':float(np.median(similarities)),
        'p95_similarity':float(np.quantile(similarities,0.95)),
        'comparisons':sorted(comparisons,key=lambda r:r['similarity'],reverse=True)}
    destination = APP_ROOT/'reports'/'semantic_similarity_audit.json'
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='comparisons'},indent=2))


if __name__ == '__main__':
    main()
