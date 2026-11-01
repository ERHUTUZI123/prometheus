"""Offline deployment invariants; does not execute research or API probes."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
errors = []
checks = {}

def check(name, condition):
    checks[name] = bool(condition)
    if not condition:
        errors.append(name)

def between(text, start, end):
    return text[text.index(start):text.index(end, text.index(start))]

old = (RUN / 'backup/AGENTS.md').read_text()
new = (ROOT / 'AGENTS.md').read_text()
check('permanent_principles_verbatim', between(old, '## 2.', '## 3.') == between(new, '## 2.', '## 3.'))
check('extra_local_evidence_rules_verbatim', between(old, '## 6.', '## 7.') == between(new, '## 6.', '## 7.'))
check('permission_boundaries_preserved', between(old, 'Digital-evolution experiments', '## 8.') == between(new, 'Digital-evolution experiments', '## 8.'))
names = ['Research Question First', 'Significance Before Tractability', 'Literature Before Novelty Claims',
         'Prevent Research Drift', 'Evidence Discipline', 'AI Collaboration', 'Stop Conditions']
check('all_local_collaboration_constraints_retained', all(new.count(x) == 1 for x in names))
check('discovery_validation_split', '### 5.1 Discovery (DWS-E)' in new and '### 5.2 Validation (DWS-V)' in new)
check('no_constitutional_branch_quota', all(x not in new for x in ('Up to 2', 'up to two', 'two distinct approaches', 'six independent', 'P/K/R')))
check('root_vision_anchor_retained', 'The Research Vision in `README.md` remains the long-term anchor.' in new)
check('single_constitution_pointer', 'not a second constitution' in (ROOT / 'deep-work/CONSTITUTION.md').read_text()
      and 'All substantive research calls must use Pro' not in (ROOT / 'deep-work/CONSTITUTION.md').read_text())

before = (RUN / 'backup/water/now.md').read_text()
after = (ROOT / 'water/now.md').read_text()
check('NOW_vision_unchanged', between(before, '## Vision', '## Current') == between(after, '## Vision', '## Current'))
check('NOW_science_body_unchanged', between(before, '## Science', '## Completed').split('\n', 1)[1]
      == between(after, '## Science', '## Completed').split('\n', 1)[1])
check('NOW_completed_history_unchanged', between(before, '## Completed', '## Deep Work') == between(after, '## Completed', '## Deep Work'))
check('one_canonical_NOW', len([p for p in (ROOT / 'water').iterdir() if p.name.lower() == 'now.md']) == 1)
check('NOW_deployment_only', '本轮只完成工作流迁移与离线验收' in after and '上一轮 Ultra 授权不延续' in after
      and '../prometheus-update/START.md' in after)

config = json.loads((ROOT / 'deep-work/runtime.example.json').read_text())
check('PKR_adjustable_entries', config['orchestration']['initial_entries'] == ['P', 'K', 'R']
      and config['orchestration']['candidate_count_is_quota'] is False)
check('fresh_R_input_and_current_mode_policy', config['orchestration']['entry_R_receives_curated_index_initially'] is False
      and config['model_policy']['requested_mode'] == 'pro'
      and config['model_policy']['previous_campaign_authorization_inherited'] is False
      and config['model_policy']['unknown_effective_remains_unknown'] is True)
check('one_main_and_distinct_backup', config['orchestration']['primary_candidates_target'] == 1
      and config['orchestration']['high_risk_alternatives_target'] == 1)
check('component_neutrality', config['selection']['new_low_level_primitives_required'] is False
      and config['selection']['no_neural_networks_requirement'] is False)
active = '\n'.join((ROOT / name).read_text() for name in ('AGENTS.md', 'deep-work/WORKFLOW.md', 'deep-work/DEPLOY.md', 'deep-work/ROLES.md'))
check('no_active_legacy_six_entry_rule', all(x not in active for x in (
    'Start six independent research tasks', '六条起始独立研究线', '初始至少四分之一', 'All substantive roles use Pro')))

json_files = sorted((ROOT / 'deep-work').rglob('*.json')) + sorted((ROOT / 'prometheus-update').rglob('*.json'))
for path in json_files:
    try:
        json.loads(path.read_text())
    except (OSError, ValueError):
        errors.append('invalid JSON: ' + str(path.relative_to(ROOT)))
checks['JSON_files_parsed'] = len(json_files)
python_files = sorted((ROOT / 'deep-work').rglob('*.py')) + sorted((ROOT / 'prometheus-update').rglob('*.py'))
for path in python_files:
    try:
        compile(path.read_text(), str(path), 'exec')
    except (OSError, SyntaxError):
        errors.append('invalid Python: ' + str(path.relative_to(ROOT)))
checks['Python_files_syntax_checked_without_execution'] = len(python_files)

package = json.loads((ROOT / 'prometheus-update/PACKAGE.json').read_text())
check('package_manifest_hashes_intact', all(
    hashlib.sha256((ROOT / 'prometheus-update' / x['path']).read_bytes()).hexdigest() == x['sha256']
    and (ROOT / 'prometheus-update' / x['path']).stat().st_size == x['bytes'] for x in package['files']))
protected = json.loads((RUN / 'protected-before.json').read_text())
changed = []
for item in protected:
    p = ROOT / item['path']
    if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != item['sha256']:
        changed.append(item['path'])
checks['protected_files_verified'] = len(protected)
check('sources_vision_old_campaigns_update_package_unchanged', not changed)
check('optional_book_navigation_unchanged', (ROOT / 'deep-work/inputs/books.md').read_bytes()
      == (RUN / 'backup/deep-work/inputs/books.md').read_bytes())

changed_paths = subprocess.run(['git', 'diff', '--name-only'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.splitlines()
check('tracked_changes_within_authorized_scope', all(x in ('AGENTS.md', 'water/now.md') or x.startswith('deep-work/') for x in changed_paths))
check('all_changed_files_backed_up', all((RUN / 'backup' / x).is_file() for x in changed_paths))
check('staged_changes_untouched', subprocess.run(['git', 'diff', '--cached', '--binary'], cwd=ROOT,
      capture_output=True, text=True, check=True).stdout == (RUN / 'git-before-staged.diff').read_text())
diff_check = subprocess.run(['git', 'diff', '--check'], cwd=ROOT, capture_output=True, text=True)
check('git_diff_check', diff_check.returncode == 0)
links = []
for name in changed_paths:
    path = ROOT / name
    if path.suffix != '.md':
        continue
    for target in re.findall(r'\]\(([^)]+)\)', path.read_text()):
        if target.startswith(('https://', 'http://', '#')):
            continue
        resolved = path.parent / target.split('#', 1)[0]
        if not resolved.exists():
            links.append({'source': name, 'target': target})
check('edited_markdown_local_links', not links)

result = {'passed': not errors, 'errors': errors, 'checks': checks, 'changed_protected_files': changed,
          'broken_links': links, 'changed_tracked_files': changed_paths,
          'scope': 'Deployment invariants, syntax, declared policy and file integrity; no research or model/API execution.'}
(RUN / 'validation.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
print(json.dumps(result, indent=2, ensure_ascii=False))
raise SystemExit(bool(errors))
