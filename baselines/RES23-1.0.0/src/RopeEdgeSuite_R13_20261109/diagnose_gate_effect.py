"""Post-run rejection ablation; changes no algorithm, record, or video output."""
from collections import Counter
import numpy as np
from common import *


def longest_run(ids):
    best = current = 0
    previous = -2
    for i in ids:
        current = current + 1 if i == previous + 1 else 1
        best = max(best, current)
        previous = i
    return best


def main():
    run = ROOT / 'runs/dual_20260923_113745_final'
    rows = read(run / 'compact.json')
    scenarios = []
    for name, removed in [
        ('actual_frozen_policy', set()),
        ('diagnostic_only_ignore_extrapolation_limit', {'unsupported_extrapolation_too_long'}),
        ('diagnostic_only_ignore_subset_sensitivity_limit', {'projected_position_sensitive_to_support_subset'}),
        ('diagnostic_only_ignore_both_limits', {'unsupported_extrapolation_too_long', 'projected_position_sensitive_to_support_subset'}),
    ]:
        ids = [r['frame'] for r in rows if not (set(r['geometry_reasons']) - removed)]
        scenarios.append(dict(name=name, removed_checks=sorted(removed),
            conditional_projection_frames=len(ids), longest_consecutive_frames=longest_run(ids)))
    measurements = []
    for i in range(600):
        g = record(i, run)['geometry_GV1']
        if g.get('extrapolation_px') is not None:
            measurements.append(g)
    quantiles = {}
    for key in ['extrapolation_px', 'observed_span_x_px', 'extrapolation_to_span_ratio', 'line_sensitivity_bound_px']:
        values = [g[key] for g in measurements if g.get(key) is not None]
        quantiles[key] = dict(zip(['p10','median','p90','max'], map(float, np.quantile(values, [.1,.5,.9,1]))))
    result = dict(source_compact_sha256=sha(run/'compact.json'), frames=600,
        reasons=dict(Counter(reason for r in rows for reason in r['geometry_reasons'])),
        scenarios=scenarios, tracking_measurement_quantiles=quantiles,
        semantics='Post-run diagnostic only. Removing a rejection is not evidence of a correct intersection or count. No production outputs changed.')
    dump(ROOT/'gate_effect_diagnostic.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
