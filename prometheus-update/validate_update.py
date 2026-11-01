#!/usr/bin/env python3
"""Validate package invariants and offline helper tests; does not deploy."""
from pathlib import Path
import json, subprocess, sys
ROOT=Path(__file__).resolve().parent
principles=[
"第一，聚焦一个可以做透的机制问题。 保留完整的 World A 愿景，但每一阶段只研究一个最有希望的因果缺口。",
"第二，用 AI 提高验证密度。 我承担大量文献核查、独立假设、反例、形式化与实验实现，让你的有限时间更多用于理解和研究判断。",
"第三，尽早让成果接受现实检验。 数学接受严格证明，计算实验接受基线与消融，研究主张接受领域同行批评。"]
a=(ROOT/'AGENTS.revised.md').read_text()
for p in principles:
    assert a.count(p)==1, 'Permanent principle missing or duplicated'
assert '### 5.1 Discovery (DWS-E)' in a
assert '### 5.2 Validation (DWS-V)' in a
for p in ROOT.rglob('*.json'):
    json.loads(p.read_text())
r=json.loads((ROOT/'deep-work/runtime.example.json').read_text())
assert r['orchestration']['initial_entries']==['P','K','R']
assert r['orchestration']['primary_candidates_target']==1
assert r['orchestration']['high_risk_alternatives_target']==1
assert r['model_policy']['previous_campaign_authorization_inherited'] is False
assert not r['selection']['new_low_level_primitives_required']
assert not r['selection']['no_neural_networks_requirement']
assert 'not a second constitution' in (ROOT/'deep-work/CONSTITUTION.md').read_text()
assert (ROOT/'INSTALL.md').is_file() and (ROOT/'START.md').is_file()
subprocess.run([sys.executable,'-m','unittest','test_tools.py','-v'],
    cwd=ROOT/'deep-work/tools',check=True)
print('Package invariants and offline tool tests passed. No deployment or research was run.')
