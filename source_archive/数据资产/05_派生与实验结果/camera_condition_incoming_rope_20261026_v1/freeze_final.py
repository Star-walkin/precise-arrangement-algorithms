"""将经过38图开发检查的代码固定后应用到400图；不将选择写成视觉验收。"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
from experiment_io import HERE, read_json, source_inventory, sha, dump

def main():
    p=argparse.ArgumentParser();p.add_argument('--pilot',type=Path,required=True)
    p.add_argument('--review',type=Path,required=True);p.add_argument('--decision',required=True)
    args=p.parse_args();run=args.pilot.absolute();review=args.review.absolute()
    out=HERE/'freeze_all.json'
    if out.exists():raise FileExistsError(out)
    if run.parent!=HERE or not run.name.startswith('pilot_'):raise ValueError('Invalid pilot directory')
    summary=read_json(run/'summary.json');audit=read_json(run/'independent_audit.json')
    context=read_json(run/'run_context.json')
    assert summary['planned']==summary['completed']==38 and summary['failed']==0
    assert summary['source_unchanged'] and summary['exit_code']==0
    assert audit['passed'] and audit['expected_images']==38 and audit['error_count']==0
    source,models=source_inventory()
    assert context['source_sha256']==source and context['model_sha256']==models
    assert review.is_file()
    paths=[run/'summary.json',run/'independent_audit.json',run/'run_context.json',review,
           HERE/'protocol.json',HERE/'evaluation_scope.json']
    dump(out,{'version':'IR-A1.0.0','selected_pilot':str(run),
              'selected_review':str(review),'decision':args.decision,
              'created_utc':datetime.now(timezone.utc).isoformat(),
              'source_sha256':source,'model_sha256':models,
              'evidence_sha256':{str(p):sha(p) for p in paths},
              'accepted_scope':'Offline 400-image review release, not contact/presence/physical accuracy acceptance',
              'core_tuning_after_freeze':False,'equipment_access':False,'reserved_inputs':False})
    print('Frozen:',run.name)

if __name__=='__main__':main()
