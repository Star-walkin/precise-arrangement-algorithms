from collections import Counter
import inspect
import pytest
from vision_geometry_lab import v24_anchor_trial as m


def fixture_index():
    ids=[f's{i}' for i in range(10)]
    entries={};slots={}
    for sid in ids:
        slots[sid]=[]
        for v in ('manual','subset0','subset1','subset2','auto'):
            key=sid+'::'+v
            entries[key]=dict(sample_id=sid,split='train',variant=v,available=True)
            slots[sid].append(dict(resolved_key=key,scheduled_variant=v,fallback_reason=None))
    return dict(train_sample_ids=ids,entries=entries,train_keys=slots)


def test_equal_base_image_exposures():
    index=fixture_index()
    a=m.training_schedule(index,'control',600,2)
    b=m.training_schedule(index,'anchor_mix',600,2)
    assert [[s['sample_id'] for s in batch] for batch in a]==[[s['sample_id'] for s in batch] for batch in b]
    assert set(Counter(s['sample_id'] for batch in b for s in batch).values())=={120}
    assert all(s['key'].endswith('::manual') for batch in a for s in batch)


def test_mix_exact_budget_per_variant():
    plan=m.training_schedule(fixture_index(),'anchor_mix',600,2)
    counts=Counter((s['sample_id'],s['scheduled_variant']) for batch in plan for s in batch)
    assert len(counts)==50 and set(counts.values())=={24}


def test_missing_auto_explicit_fallback_preserves_exposure():
    index=fixture_index();index['train_keys']['s0'][-1].update(resolved_key='s0::manual',fallback_reason='auto unavailable')
    plan=m.training_schedule(index,'anchor_mix',600,2)
    fallback=[s for batch in plan for s in batch if s['fallback_reason']]
    assert len(fallback)==24 and all(s['scheduled_variant']=='auto' and s['key']=='s0::manual' for s in fallback)


@pytest.mark.parametrize('corruption',['holdout','probe','unavailable'])
def test_bad_training_input_rejected(corruption):
    index=fixture_index();entry=index['entries']['s0::manual']
    if corruption=='holdout': entry['split']='holdout'
    elif corruption=='probe': entry['variant']='probe0'
    else: entry['available']=False
    with pytest.raises(ValueError):m.training_schedule(index,'control',1,2)


def test_inference_does_not_load_targets_or_evaluation():
    source=inspect.getsource(m.export)
    for forbidden in ('load_train_target','_dataset','build_targets','evaluate('):
        assert forbidden not in source


def test_old_checkpoint_never_saved_over():
    source=inspect.getsource(m.train)
    assert "run=base/'models'/arm" in source
    assert "torch.save" in source
    assert "p['initial_checkpoint']," not in source[source.index('torch.save'):]
