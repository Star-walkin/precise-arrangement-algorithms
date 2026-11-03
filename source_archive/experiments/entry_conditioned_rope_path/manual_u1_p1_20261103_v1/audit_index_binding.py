"""Independent render-index/record/selected-entry binding, no image generation."""
from pathlib import Path
from collections import Counter
import gzip,hashlib,json
HERE=Path(__file__).resolve().parent
def load(p):
    with (gzip.open if Path(p).suffix=='.gz' else open)(p,'rt',encoding='utf-8-sig') as f:return json.load(f)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    rendered=load(HERE/'render_receipt.json');rows=rendered['rows'];counts=Counter();conditions=0
    assert rendered['exit_code']==0 and len(rows)==len({r['id'] for r in rows})==292
    for row in rows:
        record=load(row['record_path']);assert sha(row['record_path'])==row['record_sha256']
        assert row['id']==record['id'] and row['source_sha256']==record['source_image_sha256']
        assert Path(row['source']).resolve()==Path(record['source_image']).resolve()
        assert row['selected_entry']==record['selected_entry'] and row['selected_candidate']==record['selected_candidate']
        assert row['displayed_candidates']==record['displayed_candidates'] and row['U1_model']==record['U1_model']
        assert row['entry_count']==len(record['entries'])==len(row['conditions'])
        selected=record['entries'][record['selected_entry']]
        assert Path(row['response_path']).resolve()==Path(selected['response_file']).resolve()
        assert row['response_sha256']==selected['response_sha256']
        for c,e in zip(row['conditions'],record['entries']):
            assert c['entry_index']==e['entry_index'] and c['seed_xy']==e['seed']['xy']
            assert Path(c['response_file']).resolve()==Path(e['response_file']).resolve()
            assert c['response_sha256']==e['response_sha256']
            assert c['metadata']==e['response_metadata']
            assert c['model_sha256']==record['U1_model']['sha256'] and c['model_step']==800
            assert c['source_image_sha256']==record['source_image_sha256']
            conditions+=1
        assert row['no_B_response_or_path_used'] and row['no_R1_used'] and row['no_labels_read']
        counts[row['source_use']]+=1
    assert conditions==301 and dict(counts)==rendered['source_use_counts']
    receipt=dict(exit_code=0,checked_images=292,checked_conditions=301,source_use_counts=dict(counts),
                 render_receipt_sha256=sha(HERE/'render_receipt.json'),source_sha256=sha(__file__),
                 all_render_metadata_matches_exact_saved_H6_result_and_selected_U1_condition=True,
                 no_training_prediction_or_image_generation=True,complete_identity_accuracy=None)
    out=HERE/'audit_index_binding_receipt.json'
    if out.exists():raise ValueError('Preserve prior independent binding audit')
    out.write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(receipt,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
