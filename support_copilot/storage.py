import json
import sqlite3
from datetime import datetime, timezone


class RunStore:
    def __init__(self,path):
        self.path = path
        path.parent.mkdir(parents=True,exist_ok=True)
        with self.connection() as db:
            db.executescript('''
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, brand TEXT NOT NULL,
                    route TEXT NOT NULL, status TEXT NOT NULL, reply TEXT NOT NULL,
                    review_reason TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS reviews (
                    id INTEGER PRIMARY KEY, run_id TEXT NOT NULL,
                    reviewed_at TEXT NOT NULL, decision TEXT NOT NULL,
                    reviewer TEXT NOT NULL, note TEXT NOT NULL, reply TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS outbox (
                    run_id TEXT PRIMARY KEY, created_at TEXT NOT NULL,
                    brand TEXT NOT NULL, reply TEXT NOT NULL, delivery_status TEXT NOT NULL);
            ''')

    def connection(self):
        db = sqlite3.connect(self.path,timeout=20)
        db.row_factory = sqlite3.Row
        return db

    def save(self,result):
        status = {'human_review':'pending','ready_to_send':'ready','ignored':'ignored','auto_sent':'simulated_sent'}[result['route']]
        with self.connection() as db:
            # Retries cannot overwrite a human decision for an existing run.
            inserted=db.execute('INSERT OR IGNORE INTO runs VALUES (?,?,?,?,?,?,?,?)',(
                result['run_id'],datetime.now(timezone.utc).isoformat(),result['brand'],
                result['route'],status,result.get('final_reply',''),result.get('review_reason',''),json.dumps(result,ensure_ascii=False)))
            if inserted.rowcount==1 and result['route']=='auto_sent':
                db.execute('INSERT OR IGNORE INTO outbox VALUES (?,?,?,?,?)',(
                    result['run_id'],datetime.now(timezone.utc).isoformat(),result['brand'],
                    result['final_reply'],'simulated_sent'))

    def list_runs(self,status=None,limit=100):
        with self.connection() as db:
            if status:
                rows = db.execute('SELECT * FROM runs WHERE status=? ORDER BY created_at DESC LIMIT ?',(status,limit)).fetchall()
            else:
                rows = db.execute('SELECT * FROM runs ORDER BY created_at DESC LIMIT ?',(limit,)).fetchall()
        return [dict(row) for row in rows]

    def review(self,run_id,decision,reviewer,note='',reply=None):
        if decision not in ['approved','rejected'] or not reviewer.strip():
            raise ValueError('Choose a decision and identify the reviewer.')
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM runs WHERE run_id=?',(run_id,)).fetchone()
            if not row or row['status'] != 'pending':
                raise ValueError('This review is already resolved or unavailable.')
            edited = row['reply'] if reply is None else reply
            if decision == 'approved' and (not edited.strip() or len(edited)>280):
                raise ValueError('Approval requires a nonempty reply within 280 characters.')
            if decision == 'approved':
                from .privacy import validate_reply
                payload = json.loads(row['payload'])
                issues = validate_reply(edited,[],[],payload.get('policies',[]),payload.get('cases',[]))
                if issues:
                    raise ValueError('Edit the reply before approval: ' + '; '.join(issues))
            db.execute('UPDATE runs SET status=?,reply=? WHERE run_id=?',(decision,edited,run_id))
            db.execute('INSERT INTO reviews(run_id,reviewed_at,decision,reviewer,note,reply) VALUES (?,?,?,?,?,?)',
                (run_id,datetime.now(timezone.utc).isoformat(),decision,reviewer.strip(),note,edited))
