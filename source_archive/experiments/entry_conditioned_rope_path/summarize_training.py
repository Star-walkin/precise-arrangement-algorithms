"""Post-training summaries of saved training-only metrics; no model selection."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def main(run):
    dest=run/'training_summary_stats.json';assert not dest.exists()
    r=load(run/'model/result.json');al=load(run/'model/alignment.json')
    cond=load(run/'model/condition_diagnostics/summary.json')
    stages={}
    for s in range(1,5):
        key=str(s);phase={}
        for name in ['initial_train','final_train']:
            rows=[i['stages'][key] for i in r[name]['items'] if i['stages'][key]['high_pixels']>0]
            phase[name]={'positive_images':len(rows),'macro_image_phase_balanced_mse':float(np.mean([i['balanced_mse'] for i in rows])) if rows else None}
        phase['coverage_model_pixels']={k:sum(x['stage_counts'][key][k] for x in al) for k in ['valid','high','low']}
        stages[key]=phase
    cc={key:{field:float(np.mean([c['differences'][key][field] for c in cond['items'] if c['differences'][key][field] is not None]))
        for field in ['mean_ROI','mean_far','max_far']} for key in ['zero_seed','moved_seed']}
    result={'status':'completed','exit_code':0,'steps':r['steps'],'training_seconds':r['training_seconds'],
        'stop_reason':r['stop_reason'],'overfit':{k:r['overfit'][k] for k in ['passed','steps','training_seconds','reduction','has_transition_and_attached']},
        'train_global_macro_mse':{n:r[n]['macro_image_balanced_mse'] for n in ['initial_train','final_train']},
        'train_stage_balanced_macro_mse':{n:r[n]['macro_image_stage_balanced_mse'] for n in ['initial_train','final_train']},
        'per_phase':stages,'condition_diagnostics_macro':cc,'condition_images':len(cond['items']),
        'all_moved_seeds_nonempty':all(c['moved_seed_nonempty'] for c in cond['items']),
        'scope':'training fit and condition sensitivity only; not independent identity or whole-path accuracy',
        'command':sys.argv,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();main(a.run.resolve())
