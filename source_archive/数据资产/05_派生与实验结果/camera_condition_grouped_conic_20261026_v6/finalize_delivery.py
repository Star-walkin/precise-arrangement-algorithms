"""全量完成后的独立交付检查：固定38图一致性、网页链接、源文件冻结。"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
from html.parser import HTMLParser
import json
from urllib.parse import unquote, urlparse

HERE=Path(__file__).resolve().parent

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def read(p):return json.loads(p.read_text(encoding='utf-8'))

def save(p,v):
    if p.exists():raise FileExistsError(p)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        for k,v in attrs:
            if k in ('href','src','data-smooth','data-evidence','data-group') and v:self.links.append(v)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--pilot',type=Path,required=True);ap.add_argument('--evaluation',type=Path,required=True)
    a=ap.parse_args();run=a.run;pilot=a.pilot
    audit=read(run/'integrity_audit.json');pilot_audit=read(pilot/'integrity_audit.json')
    assert audit['planned']==400 and audit['passed']==400 and audit['failed']==0 and not audit['global_errors']
    assert pilot_audit['planned']==38 and pilot_audit['passed']==38 and pilot_audit['failed']==0
    ev=read(a.evaluation/'summary.json');assert ev['predeclared_continue_criterion_met']
    context=read(run/'run_context.json');pc=read(pilot/'run_context.json')
    assert context['source_sha256']==pc['source_sha256']
    ec=read(a.evaluation/'run_context.json')
    assert ec['conic_module_sha256']==context['source_sha256'][str(HERE/'conic_family.py')]
    for path,expected in context['source_sha256'].items():
        assert sha(Path(path))==expected
        assert sha(run/'source_snapshot'/Path(path).name)==expected
    comparisons=[]
    for item in pilot_audit['file_manifest']:
        old=Path(item['path']);new=run/old.relative_to(pilot)
        h1=sha(old);h2=sha(new)
        comparisons.append({'pilot_file':str(old),'full_file':str(new),'sha256':h2,
                            'identical':h1==item['sha256']==h2})
    consistency={'created_utc':datetime.now(timezone.utc).isoformat(),'qa_images':38,
                 'compared_files':len(comparisons),'identical_files':sum(v['identical'] for v in comparisons),
                 'files':comparisons}
    save(run/'pilot_full_consistency.json',consistency)
    assert consistency['compared_files']==consistency['identical_files']==532
    pages=[HERE/'结果说明.html',run/'review.html',pilot/'review.html']
    checked=[];errors=[]
    for page in pages:
        parser=Links();parser.feed(page.read_text(encoding='utf-8'))
        for value in parser.links:
            u=urlparse(value)
            if u.scheme or not u.path:continue
            target=(page.parent/unquote(u.path)).resolve()
            good=target.is_file();checked.append({'page':str(page),'href':value,'exists':good})
            if not good:errors.append(str(target))
    links={'created_utc':datetime.now(timezone.utc).isoformat(),'checked_links':len(checked),
           'missing':errors,'checks':checked}
    save(run/'link_audit.json',links)
    assert not errors,errors
    files=[p for p in HERE.iterdir() if p.is_file() and p.name!='delivery_manifest.json']
    files += [run/n for n in ['run_context.json','run_log.jsonl','summary.json','integrity_audit.json',
                              'pilot_full_consistency.json','link_audit.json','review.html']]
    files += [pilot/n for n in ['run_context.json','summary.json','integrity_audit.json','review.html']]
    files += [a.evaluation/n for n in ['run_context.json','summary.json','predictions.jsonl','withheld_reference.jsonl']]
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'version':'RB-C3.0.0',
              'baseline':'RB-B1.0.0','selected_pilot':str(pilot),'selected_evaluation':str(a.evaluation),
              'all_run':str(run),'equipment_connected':False,'reserved_inputs':False,
              'output_file_manifest':str(run/'integrity_audit.json'),
              'files':[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files]}
    save(HERE/'delivery_manifest.json',manifest)
    print(json.dumps({'qa_files_identical':532,'web_links_checked':len(checked),'missing_links':0,
                      'delivery_manifest':str(HERE/'delivery_manifest.json')},ensure_ascii=False))

if __name__=='__main__':main()
