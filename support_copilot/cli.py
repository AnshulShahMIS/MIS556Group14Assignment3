import argparse
import json
import os

from .config import Settings
from .engine import SupportEngine
from .evaluation import evaluate
from .provider import DemoClient, ProviderClient
from .retrieval import Retriever
from .storage import RunStore


def main():
    parser=argparse.ArgumentParser(description='Support Copilot')
    commands=parser.add_subparsers(dest='command',required=True)
    commands.add_parser('build-index')
    run=commands.add_parser('run')
    run.add_argument('--brand',default='AmazonHelp')
    run.add_argument('--text',required=True)
    run.add_argument('--demo',action='store_true')
    run.add_argument('--guardrail-test',action='store_true')
    run.add_argument('--historical',action='store_true')
    benchmark=commands.add_parser('evaluate')
    benchmark.add_argument('--limit',type=int,default=200)
    benchmark.add_argument('--workers',type=int,default=4)
    benchmark.add_argument('--split',choices=['development','test'],default='test')
    commands.add_parser('bertscore')
    args=parser.parse_args()
    settings=Settings.from_env()
    os.environ.setdefault('HF_HOME',str(settings.runtime/'model_cache'))
    os.environ.setdefault('OMP_NUM_THREADS','4')
    retriever=Retriever(settings)
    if args.command=='build-index':
        def progress(n,total):
            if n%5000==0 or n==total:
                print(f'Indexed {n:,}/{total:,}',flush=True)
        print(json.dumps(retriever.build(progress),indent=2))
        return
    retriever.connect()
    if args.command=='run':
        client=DemoClient() if args.demo else ProviderClient(settings)
        result=SupportEngine(client,retriever,RunStore(settings.runtime/'runs.sqlite')).run(
            args.brand,args.text,policy_context='historical_replay' if args.historical else 'current_simulation',
            inject_bad_draft=args.guardrail_test)
        print(json.dumps(result,indent=2,ensure_ascii=False))
    elif args.command=='evaluate':
        directory,summary=evaluate(settings,retriever,args.limit,args.workers,args.split,
            progress=lambda n,total,status:print(f'Evaluated {n}/{total}: {status}',flush=True))
        print(json.dumps({'directory':str(directory),'summary':summary},indent=2))
    else:
        from .bertscore_compat import install_empty_input_compatibility
        install_empty_input_compatibility()
        from .evaluation import add_bertscore
        directories=sorted((settings.runtime/'evaluations').glob('test_*'),key=lambda p:p.stat().st_mtime,reverse=True)
        if not directories:
            raise ValueError('Run the held-out evaluation first.')
        print(json.dumps(add_bertscore(directories[0]),indent=2))


if __name__=='__main__':
    main()
