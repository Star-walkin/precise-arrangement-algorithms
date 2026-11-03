"""Read saved pilot001/002 evidence only. No detection, image tuning or labels."""
from pathlib import Path
import json,hashlib
from collections import Counter
import numpy as np

HERE=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def selected(j):
 whole=j['whole'];cs=whole.get('candidates',[]);key=whole.get('selected_candidate_id')
 return next((c for c in cs if c.get('candidate_id')==key),None)
def selected_row(j,c):
 if c is None:return None
 return next((r for r in j['occlusion'].get('candidates',[]) if r['candidate_id']==c.get('row_candidate')),None)
def detail(j):
 rows=j['occlusion'].get('rows',[]);diag=j['occlusion'].get('diagnostics',{});c=selected(j);r=selected_row(j,c)
 explicit=sum(bool(q.get('interruption_then_continuity_chain')) for q in rows)
 legacy_chain=sum(q['original_evidence_state']=='raw_continuous' and len(q.get('previous_interrupted_row_ids',[]))>=2 for q in rows)
 source=sorted(set(r.get('reference_ids',[]))) if r else []
 return {'status':j['record']['status'],'probe_count':len(rows),'states':dict(Counter(q['original_evidence_state'] for q in rows)),'distinct_visual_rows':len(set(q['visual_row_id'] for q in rows)) if rows and all('visual_row_id' in q for q in rows) else None,'near_parallel_probes':sum(q.get('near_parallel_overlap',False) for q in rows),'explicit_independent_chain_count':explicit,'legacy_two_previous_break_chain_count':legacy_chain,'retained_before_old_endpoint_filter':diag.get('retained_before_old_lower',0),'probe_budget_omitted':diag.get('truncated_rows',0),'unknown_reasons':dict(Counter(reason for q in rows for reason in q.get('unknown_reasons',[]))),'selected_candidate_id':c.get('candidate_id') if c else None,'selected_candidate_row_bonus':c.get('row_bonus') if c else None,'selected_row_candidate_id':c.get('row_candidate') if c else None,'selected_row_references':source,'selected_row_state':r.get('original_evidence_state') if r else None,'selected_row_chain_supported':bool(r.get('interruption_then_continuity_chain')) if r else False,'selected_endpoint_xy':c['xy'][-1] if c else None,'candidate_count':len(j['whole'].get('candidates',[])),'candidates_with_positive_row_bonus':sum(p.get('row_bonus',0)>0 for p in j['whole'].get('candidates',[]))}

def main():
 p1,p2=HERE/'pilot_001',HERE/'pilot_002';s1,s2=read(p1/'summary.json'),read(p2/'summary.json')
 assert s1['completed']==s2['completed']==16 and not s1['failures'] and not s2['failures']
 cases=[];totals={};state_totals={'pilot001':Counter(),'pilot002':Counter()};unknown=Counter()
 for r in s2['records']:
  rel=Path(r['output_file']);role=rel.parent.name;stem=rel.stem;f1=p1/'记录'/role/(stem+'.json');f2=p2/'记录'/role/(stem+'.json')
  a,b=read(f1),read(f2);aa,bb=detail(a),detail(b);state_totals['pilot001'].update(aa['states']);state_totals['pilot002'].update(bb['states']);unknown.update(bb['unknown_reasons'])
  ca,cb=selected(a),selected(b)
  same_path=(ca is None and cb is None) or (ca is not None and cb is not None and ca['xy']==cb['xy'] and ca['widths']==cb['widths'])
  same_row=aa['selected_row_references']==bb['selected_row_references'] and aa['selected_row_state']==bb['selected_row_state']
  cases.append({'output_file':r['output_file'],'pilot001':aa,'pilot002':bb,'same_selected_path_geometry':same_path,'same_selected_row_semantics':same_row,'row_bonus_changed':aa['selected_candidate_row_bonus']!=bb['selected_candidate_row_bonus'],'same_whole_mask_bytes':sha(p1/'01_整根候选掩膜'/role/rel.name)==sha(p2/'01_整根候选掩膜'/role/rel.name),'endpoint_delta_px':float(np.linalg.norm(np.array(ca['xy'][-1])-np.array(cb['xy'][-1]))) if ca and cb else None,'source_record_sha256':{str(f1):sha(f1),str(f2):sha(f2)}})
 for key in ('pilot001','pilot002'):
  totals[key]={name:sum(c[key][name] for c in cases) for name in ('probe_count','explicit_independent_chain_count','legacy_two_previous_break_chain_count','near_parallel_probes','retained_before_old_endpoint_filter','probe_budget_omitted','candidates_with_positive_row_bonus')}
  totals[key]['states']=dict(state_totals[key]);totals[key]['selected_positive_row_bonus_cases']=sum((c[key]['selected_candidate_row_bonus'] or 0)>0 for c in cases)
  totals[key]['selected_raw_evidence_rows']=sum(c[key]['selected_row_state'] in ('raw_continuous','raw_interruption_supported') for c in cases)
  totals[key]['selected_chain_supported_cases']=sum(c[key]['selected_row_chain_supported'] for c in cases)
 report={'scope':'Read-only saved fixed16 comparison after predictions, no labels or model execution. Probe counts are not independent physical rows or accuracy.','cases_count':16,'totals':totals,'changed_selected_geometry_cases':sum(not c['same_selected_path_geometry'] for c in cases),'changed_selected_row_semantics_cases':sum(not c['same_selected_row_semantics'] for c in cases),'changed_selected_row_bonus_cases':sum(c['row_bonus_changed'] for c in cases),'identical_whole_mask_byte_cases':sum(c['same_whole_mask_bytes'] for c in cases),'pilot002_unknown_reasons':dict(unknown),'raw_but_nonchain_does_not_establish_n_minus_one':True,'interpretation':'Original continuity/interruption of an individual row is local image evidence only. n-1-like evidence additionally needs consecutive distinct interrupted crossings followed by a continuous crossing. Even that is not confirmed same-material identity. All curved continuations remain review candidates in pilot002.','algorithm_source_sha256':{str(p):sha(p) for p in [HERE/'boundary_occlusion.py',HERE/'boundary_occlusion_revision2.py',p1/'source_snapshot/boundary_occlusion.py',p2/'source_snapshot/boundary_occlusion.py']},'cases':cases}
 out=p2/'boundary_evidence_comparison.json'
 if out.exists():raise ValueError('Never overwrite comparison report')
 out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
 print(json.dumps({k:v for k,v in report.items() if k not in ('cases','algorithm_source_sha256','pilot002_unknown_reasons')},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
