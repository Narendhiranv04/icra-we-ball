#!/usr/bin/env python3
"""Run one clean live matrix only after non-semantic endpoint/git preflight.

Does not start, stop, or configure vLLM. No semantic request during preflight.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,default=Path('benchmark_reports/final_vlm_evaluation_v3'))
    parser.add_argument('--base-url',default='http://127.0.0.1:8000/v1')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    os.chdir(root)
    sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','status','--porcelain'],text=True).strip()
    if dirty:
        raise SystemExit('Preflight blocked: working tree is not clean')
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit('Preflight blocked: output directory already contains artifacts')
    try:
        with urllib.request.urlopen(args.base_url.rstrip('/')+'/models',timeout=10) as response:
            models=json.load(response)['data']
    except Exception as exc:
        raise SystemExit(f'Preflight blocked: existing vLLM endpoint unavailable: {exc}')
    ids=[m['id'] for m in models]
    if not ids:
        raise SystemExit('Preflight blocked: endpoint exposes no models')
    model='qwen35-9b' if 'qwen35-9b' in ids else ids[0]
    # The schema version has to be pinned here.  The replay drivers set it and
    # this one did not, so a live matrix launched without it in the environment
    # silently ran the V2 contract against the V3 pipeline and scored zero
    # across the board -- which looks like a catastrophic result and is only a
    # missing variable.
    env=dict(os.environ,TAMP_FM_BASE_URL=args.base_url,TAMP_FM_MODEL=model,
             TAMP_FM_MAX_TOKENS='24000',TAMP_FM_SCHEMA_VERSION='3',PYTHONPATH='.')
    command=[sys.executable,'scripts/evaluate_vlm_functional_tamp.py','--mode','vlm','--spec-source','live','--output-root',str(args.output_root)]
    print(f'Frozen commit: {sha}; model: {model}; schema: v3; '
          f'max_tokens: 24000; output: {args.output_root}',flush=True)
    result=subprocess.run(command,env=env)
    after_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    after_dirty=subprocess.check_output(['git','status','--porcelain'],text=True).strip()
    errors=[]
    if result.returncode:errors.append(f'Evaluator exit code {result.returncode}')
    if after_sha!=sha or after_dirty:errors.append('Code/working tree changed during matrix')
    if args.output_root.exists():
        inv_path=args.output_root/'invariants.json'
        invariants=json.loads(inv_path.read_text()) if inv_path.exists() else {'errors':['Matrix did not finish']}
        errors.extend(invariants.get('errors',[]))
        inv_path.write_text(json.dumps({'status':'INVALID' if errors else 'VALID','errors':errors,
            'pre_evaluation_sha':sha,'post_evaluation_sha':after_sha,'working_tree_clean':not after_dirty,
            'actual_model_id':model},indent=2)+'\n')
    return 1 if errors else 0


if __name__=='__main__':
    raise SystemExit(main())
