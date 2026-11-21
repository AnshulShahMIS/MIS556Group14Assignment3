"""Audit TWCS and prepare disjoint conversation groups for Support Copilot."""

import argparse
import csv
import hashlib
import json
import re
import sqlite3
import sys
import time
from array import array
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path


BRANDS = ['AmazonHelp', 'AppleSupport', 'Uber_Support', 'SpotifyCares', 'Delta']


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def clean_text(text):
    # Keep brand separately in metadata. Redactions preserve issue wording.
    text = re.sub(r'https?://\S+', '[URL]', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', '[EMAIL]', text)
    text = re.sub(r'(?<!\w)@[A-Za-z0-9_]+', '[USER]', text)
    text = re.sub(r'(?<!\w)(?:\+?\d[\d ()-]{7,}\d)(?!\w)', '[NUMBER]', text)
    return re.sub(r'\s+', ' ', text).strip()


def normalized_query(text):
    text = clean_text(text).lower()
    text = re.sub(r'\[[a-z]+\]', ' ', text)
    return re.sub(r'[^\w]+', ' ', text).strip()


def timestamp(value):
    return int(parsedate_to_datetime(value).timestamp())


def initialize_language_worker(dependency_path):
    if dependency_path:
        sys.path.insert(0, dependency_path)
    import langid
    langid.classify('Initialize the language identification model.')


def detect_pair_language(texts):
    import langid
    customer, reply = texts
    customer_language, _ = langid.classify(re.sub(r'\[[A-Z]+\]', '', customer))
    if customer_language != 'en':
        return 'non_english_customer'
    reply_language, _ = langid.classify(re.sub(r'\[[A-Z]+\]', '', reply))
    return 'en' if reply_language == 'en' else 'non_english_reply'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--sample', required=True)
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--dependency-path')
    parser.add_argument('--index-size', type=int, default=50000)
    parser.add_argument('--max-id', type=int, default=2987950)
    args = parser.parse_args()
    if args.dependency_path:
        sys.path.insert(0, args.dependency_path)

    source = Path(args.source)
    work = Path(args.workspace)
    out = Path(args.output)
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'data').mkdir(exist_ok=True)
    size = args.max_id + 1
    parent = array('I', range(size))
    weight = array('I', [1]) * size
    present = bytearray(size)

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left, right):
        if max(left, right) >= len(parent):
            old = len(parent)
            extension = max(left, right) + 1 - old
            parent.extend(range(old, old + extension))
            weight.extend([1] * extension)
            present.extend(bytes(extension))
        left, right = find(left), find(right)
        if left == right:
            return
        if weight[left] < weight[right]:
            left, right = right, left
        parent[right] = left
        weight[left] += weight[right]

    with Path(args.sample).open(encoding='utf-8-sig', newline='') as stream:
        samples = {int(r['tweet_id']): r for r in csv.DictReader(stream)}
    samples_by_text = {r['text']:tweet_id for tweet_id,r in samples.items()}
    sample_text_matches = 0
    sample_text_and_id_matches = 0
    sample_id_examples, sample_text_examples = [], []
    audit = Counter()
    outbound = Counter()
    db = sqlite3.connect(work / 'pair_candidates.sqlite')
    db.executescript('''
        DROP TABLE IF EXISTS roots;
        DROP TABLE IF EXISTS replies;
        CREATE TABLE roots (id INTEGER PRIMARY KEY, created INTEGER, text TEXT, declared_replies TEXT);
        CREATE TABLE replies (id INTEGER PRIMARY KEY, parent_id INTEGER, brand TEXT, created INTEGER, text TEXT);
    ''')
    root_batch, reply_batch = [], []
    began = time.time()
    with source.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        for row in reader:
            tweet_id = int(row['tweet_id'])
            union(tweet_id, tweet_id)
            audit['duplicate_tweet_ids'] += int(present[tweet_id])
            present[tweet_id] = 1
            inbound = row['inbound'].lower() == 'true'
            audit['tweets'] += 1
            audit['inbound' if inbound else 'outbound'] += 1
            parent_id = int(row['in_response_to_tweet_id']) if row['in_response_to_tweet_id'] else None
            if parent_id is not None:
                union(tweet_id, parent_id)
            if row['response_tweet_id']:
                for reply_id in row['response_tweet_id'].split(','):
                    union(tweet_id, int(reply_id.strip()))
            elif inbound:
                audit['inbound_without_listed_response'] += 1
            if ',' in row['response_tweet_id']:
                audit['tweets_with_multiple_listed_responses'] += 1
            if inbound and parent_id is None:
                root_batch.append((tweet_id, timestamp(row['created_at']), row['text'], row['response_tweet_id']))
                audit['root_customer_tweets'] += 1
            if not inbound:
                outbound[row['author_id']] += 1
                if row['author_id'] in BRANDS and parent_id is not None:
                    reply_batch.append((tweet_id, parent_id, row['author_id'], timestamp(row['created_at']), row['text']))
            if tweet_id in samples:
                audit['sample_ids_found_in_full'] += 1
                audit['sample_text_mismatches'] += int(row['text'] != samples[tweet_id]['text'])
                if len(sample_id_examples) < 3:
                    sample_id_examples.append({'id':str(tweet_id),
                        'same_author':row['author_id']==samples[tweet_id]['author_id'],
                        'same_text':row['text']==samples[tweet_id]['text']})
            if row['text'] in samples_by_text:
                sample_text_matches += 1
                sample_text_and_id_matches += int(samples_by_text[row['text']] == tweet_id)
                if len(sample_text_examples) < 5:
                    sample_text_examples.append({'sample_id':str(samples_by_text[row['text']]),'full_id':str(tweet_id)})
            if len(root_batch) >= 10000:
                db.executemany('INSERT INTO roots VALUES (?,?,?,?)', root_batch)
                root_batch.clear()
            if len(reply_batch) >= 10000:
                db.executemany('INSERT INTO replies VALUES (?,?,?,?,?)', reply_batch)
                reply_batch.clear()
            if audit['tweets'] % 500000 == 0:
                print(f"Read {audit['tweets']:,} tweets in {time.time()-began:.0f}s", flush=True)
    db.executemany('INSERT INTO roots VALUES (?,?,?,?)', root_batch)
    db.executemany('INSERT INTO replies VALUES (?,?,?,?,?)', reply_batch)
    db.commit()
    db.execute('CREATE INDEX reply_parent ON replies(parent_id)')
    db.commit()

    # Earliest direct brand response, with deterministic tie-breaking. Multiple
    # brands may reply to a root, so brand stays part of the candidate identity.
    query = '''
        WITH ranked AS (
            SELECT r.id AS customer_id, r.created AS customer_created,
                   r.text AS customer_text, r.declared_replies,
                   s.id AS reply_id, s.brand, s.created AS reply_created, s.text AS reply_text,
                   ROW_NUMBER() OVER(PARTITION BY r.id, s.brand ORDER BY s.created, s.id) AS n
            FROM roots r JOIN replies s ON s.parent_id=r.id
            WHERE s.created>=r.created
        ) SELECT * FROM ranked WHERE n=1 ORDER BY customer_created, customer_id, brand
    '''
    db.row_factory = sqlite3.Row
    candidates = list(db.execute(query))
    candidate_counts = Counter(r['brand'] for r in candidates)
    print('Confirmed root-to-brand candidates ' + json.dumps(candidate_counts), flush=True)
    group_sizes = Counter()
    for tweet_id, is_present in enumerate(present):
        if is_present:
            group_sizes[find(tweet_id)] += 1
    group_min_id = {}
    for tweet_id, is_present in enumerate(present):
        if is_present:
            group_min_id.setdefault(find(tweet_id), tweet_id)

    language_counts = Counter()
    seen_queries = set()
    english_by_brand = {brand: [] for brand in BRANDS}
    # Round-robin chronological windows avoid filling the corpus with one brand.
    buckets = {brand: [] for brand in BRANDS}
    for row in candidates:
        buckets[row['brand']].append(row)
    # Deterministic broad sample across each brand's date range. Every selected
    # record is still assigned as a whole connected conversation in the split.
    max_per_brand = args.index_size // len(BRANDS) + 2500
    with ProcessPoolExecutor(max_workers=4, initializer=initialize_language_worker,
                             initargs=(args.dependency_path,)) as pool:
        for brand in BRANDS:
            bucket = buckets[brand]
            indices = sorted(range(len(bucket)), key=lambda i: hashlib.sha256(f"twcs-v1-{bucket[i]['customer_id']}-{brand}".encode()).digest())
            for offset in range(0, len(indices), 1000):
                if len(english_by_brand[brand]) >= max_per_brand:
                    break
                batch = []
                for i in indices[offset:offset+1000]:
                    row = bucket[i]
                    customer_text, reply_text = clean_text(row['customer_text']), clean_text(row['reply_text'])
                    query_key = normalized_query(customer_text)
                    if len(query_key.split()) < 4:
                        language_counts['short_or_empty_customer_text'] += 1
                        continue
                    if query_key in seen_queries:
                        language_counts['duplicate_normalized_query'] += 1
                        continue
                    batch.append((row,customer_text,reply_text,query_key))
                languages = pool.map(detect_pair_language, [(x[1],x[2]) for x in batch], chunksize=100)
                for (row,customer_text,reply_text,query_key), language in zip(batch,languages):
                    if language != 'en':
                        language_counts[language] += 1
                        continue
                    if query_key in seen_queries:
                        language_counts['duplicate_normalized_query'] += 1
                        continue
                    if len(english_by_brand[brand]) >= max_per_brand:
                        continue
                    seen_queries.add(query_key)
                    component = find(row['customer_id'])
                    declared = {int(v.strip()) for v in row['declared_replies'].split(',') if v.strip()}
                    english_by_brand[brand].append({
                        'case_id': f"twcs-{row['customer_id']}-{row['reply_id']}",
                        'conversation_id': f"component-{group_min_id[component]}",
                        'conversation_tweet_count': group_sizes[component],
                        'brand': brand, 'customer_tweet_id': str(row['customer_id']),
                        'reply_tweet_id': str(row['reply_id']),
                        'created_at': datetime.fromtimestamp(row['customer_created'], timezone.utc).isoformat(),
                        'reply_created_at': datetime.fromtimestamp(row['reply_created'], timezone.utc).isoformat(),
                        'customer_text': customer_text, 'historical_reply': reply_text,
                        'language': 'en', 'language_method': 'langid classification on both redacted texts',
                        'normalized_query_hash': hashlib.sha256(query_key.encode()).hexdigest(),
                        'reciprocal_response_link': row['reply_id'] in declared,
                        'source': 'thoughtvector/customer-support-on-twitter',
                        'redaction_status': 'regex redaction of handles, URLs, email addresses and long number patterns; manual review still needed'
                    })
                    if len(english_by_brand[brand]) % 2500 == 0:
                        print(f"English pairs for {brand}: {len(english_by_brand[brand]):,}", flush=True)

    # Assign connected groups globally before pair caps. This prevents a reply or
    # another brand's part of a conversation from crossing retrieval/test sets.
    group_split = {}
    for records in english_by_brand.values():
        for row in records:
            group = row['conversation_id']
            value = int(hashlib.sha256(('split-v1-' + group).encode()).hexdigest()[:8], 16) % 100
            group_split[group] = 'test' if value < 5 else ('development' if value < 10 else 'index')
    splits = {'index': [], 'development': [], 'test': []}
    targets = {'index': args.index_size // 5, 'development': 20, 'test': 40}
    for brand in BRANDS:
        records = sorted(english_by_brand[brand], key=lambda r: r['created_at'])
        for split in splits:
            matching = [r for r in records if group_split[r['conversation_id']] == split]
            if len(matching) < targets[split]:
                raise ValueError(f'Insufficient {split} rows for {brand}: {len(matching)}')
            splits[split].extend(matching[:targets[split]])
    for split, rows in splits.items():
        with (out / 'data' / f'{split}_pairs.jsonl').open('w', encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    for left, right in [('index','test'), ('index','development'), ('development','test')]:
        assert not ({r['conversation_id'] for r in splits[left]} & {r['conversation_id'] for r in splits[right]})
        assert not ({r['normalized_query_hash'] for r in splits[left]} & {r['normalized_query_hash'] for r in splits[right]})
    # Keep reference replies outside the inputs sent to any evaluated system.
    with (out / 'data' / 'test_inputs.jsonl').open('w', encoding='utf-8') as stream:
        for row in splits['test']:
            stream.write(json.dumps({k:row[k] for k in ['case_id','brand','created_at','customer_text']},ensure_ascii=False)+'\n')
    annotation_columns = ['case_id', 'brand', 'customer_text', 'intent', 'urgency', 'needs_human', 'escalation_reason', 'annotator']
    with (out / 'data' / 'triage_annotation_template.jsonl').open('w',encoding='utf-8') as stream:
        for row in splits['test'][::2]:
            record = {key: row.get(key,'') for key in annotation_columns}
            stream.write(json.dumps(record,ensure_ascii=False)+'\n')
    sample = list(csv.DictReader(Path(args.sample).open(encoding='utf-8-sig',newline='')))
    report = {
        'source_file': str(source), 'source_bytes': source.stat().st_size,
        'source_sha256': hashlib.file_digest(source.open('rb'),'sha256').hexdigest(),
        'columns': columns, 'counts': dict(audit), 'support_accounts': len(outbound),
        'top_support_accounts': outbound.most_common(20),
        'sample_rows': len(sample), 'sample_overlap': audit['sample_ids_found_in_full'],
        'selected_brands': BRANDS, 'confirmed_first_reply_pairs_by_brand': dict(candidate_counts),
        'observed_conversation_components': len(group_sizes),
        'components_by_observed_tweet_count': dict(Counter(group_sizes.values())),
        'language_and_dedup_exclusions_in_sampled_pool': dict(language_counts),
        'language_filter_limitations': 'Automatic detection on short tweets is imperfect. This is a filtered sampled pool, not an exhaustive language audit.',
        'split_method': 'Deterministic SHA256 assignment of connected components, exact normalized-query deduplication, then per-brand caps.',
        'split_counts': {split:len(rows) for split,rows in splits.items()},
        'split_brand_counts': {split:dict(Counter(r['brand'] for r in rows)) for split,rows in splits.items()},
        'split_time_ranges': {split:{'start':min(r['created_at'] for r in rows),'end':max(r['created_at'] for r in rows)} for split,rows in splits.items()},
        'split_checks': {'conversation_overlap':0,'normalized_query_overlap':0,'near_duplicate_semantic_check':'pending'},
        'policy_data': 'TWCS contains historical responses. Brand-approved policies, account facts, VIP status and resolution outcomes are not provided.',
        'evaluation_status': 'Prepared inputs and historical references. No triage labels, agent results, quality gains, BERTScores or escalation metrics have been measured.',
        'elapsed_seconds':round(time.time()-began,1)
    }
    save_json(out / 'data_audit.json', report)
    save_json(out / 'sample_compatibility_audit.json', {
        'sample_rows':len(samples), 'matching_text_rows':sample_text_matches,
        'matching_text_and_same_id':sample_text_and_id_matches,
        'id_examples':sample_id_examples,'text_id_examples':sample_text_examples})
    db.close()
    print(json.dumps({k:report[k] for k in ['counts','support_accounts','sample_rows','sample_overlap','confirmed_first_reply_pairs_by_brand','split_counts','split_checks','elapsed_seconds']},indent=2),flush=True)


if __name__ == '__main__':
    main()
