"""Dataset integrity, author metadata, workflow and paper/result consistency."""
from pathlib import Path
import hashlib,json,re
import numpy as np
import yaml
root=Path(__file__).resolve().parents[1]
for name,expected in json.loads((root/'data/SHA256SUMS.json').read_text()).items():
 assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
cff=yaml.safe_load((root/'CITATION.cff').read_text())
assert cff['authors']==[{'family-names':'Khanna','given-names':'Tarang'}]
assert cff['license']=='MIT' and 'doi' not in cff
assert (root/'LICENSE').read_text().startswith('MIT License')
assert cff['repository-code']=='https://github.com/TarangKhanna/cached-rollout-verification'
wf=yaml.load((root/'.github/workflows/tests.yml').read_text(),Loader=yaml.BaseLoader)
assert set(wf['on'])=={'push','pull_request','workflow_dispatch'}
assert wf['permissions']=={'contents':'read'}
assert any('bash run_checks.sh' in s.get('run','') for s in wf['jobs']['checks']['steps'])
assert all(re.fullmatch(r'[^@]+@[0-9a-f]{40}',s['uses']) for s in wf['jobs']['checks']['steps'] if 'uses' in s)
tex=(root/'manuscript.tex').read_text();readme=(root/'README.md').read_text()
assert r'\author{Tarang Khanna}' in tex
assert cff['title']=='A Conditional-Proposal Pitfall in Cached-Rollout Speculative Verification: Worked Examples and Controls'
assert 'not a new RL algorithm' in readme
for fraction in ['1/8','7/32','7/64']:
 assert fraction in tex and fraction in readme
assert 'OpenAI systems, including ChatGPT and Codex' in tex
keys=set(k for cs in re.findall(r"\\cite[a-z]*(?:\[[^\]]*\])*\{([^}]+)\}",tex) for k in cs.split(','))
bib=set(re.findall(r"@\w+\{([^,]+),",(root/'references.bib').read_text()))
assert keys and keys<=bib
for m in re.findall(r'\]\(([^)]+)\)',readme):
 if not m.startswith(('https://','http://','#')):assert (root/m.split('#')[0]).exists(),m
rows=[json.loads(s) for s in (root/'data/batch/qualification.jsonl').read_text().splitlines()]
for eta,val in [(.05,.0249),(.2,.0848)]:
 c=next(x['results'] for x in rows if x['type']=='cell' and x['results']['B']==128 and x['results']['eta']==eta)
 assert round(100*c['relative_upper95'],4)==val
 assert str(val) in (root/'REPRODUCIBILITY.md').read_text() and str(val) in tex
details=(root/'REPRODUCIBILITY.md').read_text()
neural=json.loads((root/'data/readout/results.json').read_text())['results'][-1]
assert round(100*neural['contrasts']['C-B']['projected_relative_bias'],4)==-1.4475
assert '-1.4475' in details and '-1.4475' in tex
print('PASS dataset hashes, sole-author citation, pinned read-only CI, paper/README numerical agreement and local links')
