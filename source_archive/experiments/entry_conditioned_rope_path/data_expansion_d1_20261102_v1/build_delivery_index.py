"""Add explicit source-use metadata without modifying audited PNGs/receipts."""
from pathlib import Path
from collections import Counter
import json
import hashlib

HERE = Path(__file__).resolve().parent

def read(name):
    return json.loads((HERE / name).read_text(encoding='utf-8-sig'))

def main():
    protocol = read('protocol.json')
    data = read('dataset/index.json')
    rendered = read('render_receipt.json')
    assert len(rendered['rows']) == 292
    assert len({x['id'] for x in rendered['rows']}) == 292
    training = {x['image_sha256']: x for x in data['items']}
    train_groups = set(protocol['train_groups'])
    eval_groups = set(protocol['eval_groups'])
    assert not train_groups & eval_groups
    rows = []
    for x in rendered['rows']:
        matched = training.get(x['source_sha256'])
        assert bool(matched) == x['source_is_D1_training_image']
        if matched:
            use = 'D1_training_image_' + matched['origin']
        elif x['source_group'] in eval_groups:
            use = 'historical_open_evaluation_source_group_display'
        elif x['source_group'] in train_groups:
            use = 'same_training_source_group_unused_frame_display'
        else:
            use = 'other_open_untrained_source_group_display'
        rows.append(dict(
            source_id=x['id'], source_group=x['source_group'], source_use=use,
            independent_blind_test=False, source=x['source'], source_sha256=x['source_sha256'],
            output=x['output'], output_sha256=x['sha256'], model=x['model'],
            model_step=x['model_step'], entry_index=x['display_entry'],
            entry_count=x['entry_count'], entry_source=x['entry_source'],
            entry_choice='pre-inference frozen index0; no best-entry selection',
            response_path=x['response_path'], conditions=x['conditions'],
            source_is_D1_training_image=bool(matched),
            matched_training_id=matched['id'] if matched else None,
            inference_status=x['status'], error_reason=None if x['status']=='predicted' else x['status']))
    result = dict(
        date_local='2026-11-03', unique_source_count=292,
        entry_conditions=sum(x['entry_count'] for x in rows),
        source_use_counts=dict(Counter(x['source_use'] for x in rows)),
        render_receipt_sha256=hashlib.sha256((HERE/'render_receipt.json').read_bytes()).hexdigest(),
        note='PNG footers distinguish actual training images from non-training display images. This supplemental index further separates historical evaluation groups, unused training-group frames, and other sources. No PNG, response, label or original receipt changed.',
        rows=rows)
    assert result['entry_conditions'] == 301
    (HERE/'final_292_index.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
