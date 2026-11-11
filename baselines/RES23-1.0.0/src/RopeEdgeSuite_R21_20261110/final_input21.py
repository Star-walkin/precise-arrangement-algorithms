from common21 import *

def compose(item):
    id=item['id'];rows=read(ROOT/'measurements'/id/'frames.json');replacements=0
    for r in rows:
        i=r['frame'];bp=ROOT/'dynamic_b'/id/'frames'/f'{i:06d}.json.gz'
        path=bp if r['fixed_d'] is None else ROOT/'measurements'/id/'frames'/f'{i:06d}.json.gz'
        if r['fixed_d'] is None:
            row=gzread(path);m=row['measurement'];c=m.get('central') or {};b=c.get('boundary') or {}
            r.update(available=m['available'],N=m.get('N'),N_interval=m.get('N_interval'),d=m.get('d'),d_interval=m.get('d_interval'),
                measurement_mode=m['measurement_mode'],complete_F3_phase=m.get('complete_F3_phase',False),observed_grooves=m.get('observed_groove_count'),
                pitch=c.get('local_pitch_px'),boundary_occluded=b.get('occluded'),boundary_local=b.get('locally_observed'),
                boundary_uncertainty_px=b.get('uncertainty_px'),boundary_anchor_hull=b.get('inside_current_anchor_hull'),
                contact_gap=c.get('contact_projected_gap_N'),reasons=m['reasons'])
            replacements+=1
        r['measurement_record_path']=str(path);r['measurement_record_sha256']=sha(path)
    dump(ROOT/'final_measurements'/id/'frames.json',rows)
    dump(ROOT/'final_measurements'/id/'summary.json',dict(id=id,frames=len(rows),available=sum(x['available'] for x in rows),
        dynamic=sum(x['measurement_mode']=='current_dynamic_column' for x in rows),dynamic_gate_B_frames=replacements))
    return rows
