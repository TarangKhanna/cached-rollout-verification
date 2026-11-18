"""Saved-data-only bootstrap recomputation extracted from the archived verifier.
No model, backbone inference, new rollout experiment or full-policy training.
"""
from pathlib import Path
import json
import hashlib
import numpy as np
OUT=Path(__file__).resolve().parents[1]/'data/readout'

def main():
    result=json.loads((OUT/'results.json').read_text());cal=json.loads((OUT/'calibration.json').read_text())
    metrics=json.loads((OUT/'replica_metrics.json').read_text())
    assert hashlib.sha256((OUT/'calibration.json').read_bytes()).hexdigest()==result['calibration_sha256']
    assert len(metrics)==4 and all(len(x)==256 for x in metrics)
    rng=np.random.default_rng(2026111703);idx=rng.integers(256,size=(2000,256))
    comparisons=0;raw_primary=[]
    for level,rows,saved in zip(cal['levels'],metrics,result['results']):
        v=np.array([r['norm2'] for r in rows]);vb=v[idx].mean(axis=1)
        for name,estimate in saved['contrasts'].items():
            raw=np.array([r['numerators'][name] for r in rows]);c=np.array([r['controls'][name] for r in rows])
            k=cal['kappa'][level['label']][name];adjusted=raw-k*c
            ci=np.quantile(adjusted[idx].mean(axis=1)/vb,[.025,.975])
            np.testing.assert_allclose(ci,estimate['relative_95'],atol=1e-13)
            np.testing.assert_allclose(adjusted.mean()/v.mean(),estimate['projected_relative_bias'],atol=1e-13)
            comparisons+=1
            if name=='C-B':raw_primary.append({'level':level['label'],
                'raw_relative_95':np.quantile(raw[idx].mean(axis=1)/vb,[.025,.975]).tolist()})
    assert comparisons==24
    print('PASS 24 archived readout ratio-bootstrap contrasts (four levels, 256 update replicas)')
    print('Saved-data statistics only: no inference, seed replay or dense gradient reproduction')
    print(json.dumps({'highest_KL': result['results'][-1]['mean_actor_KL'],
                      'C-B': result['results'][-1]['contrasts']['C-B']},sort_keys=True))

if __name__=='__main__': main()
