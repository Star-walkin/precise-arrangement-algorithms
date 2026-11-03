"""开发对照结束后冻结一次；记录失败，不把全量输出等同验收。"""
from datetime import datetime,timezone
from experiment_io import HERE,read_json,dump,sha,sources

def main():
    target=HERE/'freeze_all.json'
    if target.exists():raise FileExistsError(target)
    pilot=HERE/'pilot_003_global_gamma'
    summary=read_json(pilot/'summary.json')
    audit=read_json(pilot/'independent_audit.json')
    context=read_json(pilot/'run_context.json')
    if summary['completed']!=49 or summary['failed'] or not summary['source_unchanged']:raise ValueError('Pilot incomplete')
    if context['source_sha256']!=sources():raise ValueError('Source drift')
    if audit['passed']!=49 or audit['failed'] or audit['global_errors']:raise ValueError('Audit failed')
    visual=HERE/'freeze_visual_notes.json'
    if not visual.exists():raise ValueError('Visual release decision missing')
    dump(target,{'version':'IR-A2.0.0','created_utc':datetime.now(timezone.utc).isoformat(),
        'variant':'global_gamma','gamma':.70,'source_sha256':sources(),
        'pilot_run':pilot.name,'pilot_summary_sha256':sha(pilot/'summary.json'),
        'pilot_audit_sha256':sha(pilot/'independent_audit.json'),
        'visual_release_decision':visual.name,'visual_review_sha256':sha(visual),'evaluation_scope_sha256':sha(HERE/'evaluation_scope.json'),
        'decision':'Export fixed400 with milder global gamma; stop score tuning after three core versions.',
        'reason':'Illumination-field gain amplified color block noise; global gamma preserves a milder display. Increased C3 counts do not establish accuracy.',
        'known_failure':'Parallel offset and wrong neighboring-coil joins persist. No same-rope or contact acceptance. Candidate paths are diagnostic.',
        'release_type':'offline_experimental_results_not_accepted_baseline','replace_C3_or_A1':False,
        'equipment_access':False,'reserved_inputs':False})
    print(target)

if __name__=='__main__':main()
