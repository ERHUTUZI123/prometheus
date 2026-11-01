#!/usr/bin/env python3
"""Local offline checks; makes no API call and runs no scientific experiment."""
from __future__ import annotations
import copy
import json
import tempfile
import unittest
from pathlib import Path
from audit import audit
from pro_probe import make_receipt, make_request, receipt_from


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for file in ['receipt.json', 'a.md', 'b.md', 'inputs.json', 'result.md']:
            (self.root / file).write_text('fixture', encoding='utf-8')
        (self.root / 'authorization.json').write_text(json.dumps({
            'run_id': 'test', 'requested_mode': 'pro',
            'authorization_scope': 'this_run',
            'authorized_observed_modes': ['pro', 'ultra'],
        }), encoding='utf-8')
        self.mode = {'model':'gpt-6-astra','requested_mode':'pro','effective_mode':'pro',
                     'evidence_kind':'provider_metadata','receipt_path':'receipt.json'}
        self.record = {'run_id':'test','status':'completed','mode_evidence':copy.deepcopy(self.mode),
                       'tasks':[{'task_id':'a','status':'completed','depends_on':[],
                                 'mode_evidence':copy.deepcopy(self.mode),'output_path':'a.md',
                                 'input_manifest':'inputs.json','independent_branch':True,
                                 'context_id':'context-a','prior_report_seen_before_freeze':False,
                                 'isolation_kind':'runtime_enforced'}],
                       'probes':[], 'final_report':'result.md'}
    def tearDown(self):
        self.tmp.cleanup()
    def test_valid(self):
        self.assertFalse(audit(self.record,self.root,True)[0])
    def test_high_not_pro(self):
        self.record['tasks'][0]['mode_evidence']['effective_mode']='standard'
        self.assertTrue(audit(self.record,self.root)[0])
    def test_mode_unknown(self):
        self.record['mode_evidence']['effective_mode']=None
        self.assertTrue(audit(self.record,self.root)[0])
    def test_shared_context(self):
        t=copy.deepcopy(self.record['tasks'][0]);t['task_id']='b';t['output_path']='b.md'
        self.record['tasks'].append(t)
        self.assertTrue(audit(self.record,self.root)[0])
    def test_missing_output(self):
        self.record['tasks'][0]['output_path']='missing.md'
        self.assertTrue(audit(self.record,self.root)[0])
    def test_no_escape(self):
        self.record['tasks'][0]['output_path']='../result.md'
        self.assertTrue(audit(self.record,self.root)[0])
    def test_dependency_cycle(self):
        self.record['tasks'][0]['depends_on']=['a']
        self.assertTrue(audit(self.record,self.root)[0])
    def test_ui_warning(self):
        self.record['tasks'][0]['mode_evidence'].update(evidence_kind='ui_user_attested', effective_mode=None, observed_mode='pro')
        errors,warnings=audit(self.record,self.root)
        self.assertFalse(errors);self.assertTrue(any('attestation' in w for w in warnings))
    def test_old_report_seen(self):
        self.record['tasks'][0]['prior_report_seen_before_freeze']=True
        self.assertTrue(audit(self.record,self.root)[0])
    def test_probe_request(self):
        r=make_request('gpt-6-astra','max')
        self.assertEqual(r['reasoning'],{'mode':'pro','effort':'max'})
    def test_probe_evidence_is_not_request(self):
        r=receipt_from({'model':'gpt-6-astra'}, make_request('gpt-6-astra','max'))
        self.assertFalse(r['pro_verified']);self.assertEqual(r['evidence_kind'],'requested_only')
    def test_probe_verified(self):
        r=receipt_from({'model':'gpt-6-astra','reasoning':{'mode':'pro'},'status':'completed'},make_request('gpt-6-astra','max'))
        self.assertTrue(r['pro_verified'])

    def test_host_selected_pro_effective_unknown(self):
        self.record['tasks'][0]['mode_evidence'].update(
            evidence_kind='host_metadata', effective_mode=None, observed_mode='pro')
        errors,warnings=audit(self.record,self.root)
        self.assertFalse(errors);self.assertTrue(any('unknown' in w for w in warnings))
    def test_host_cannot_claim_provider_effective(self):
        self.record['tasks'][0]['mode_evidence'].update(
            evidence_kind='host_metadata', effective_mode='pro', observed_mode='pro')
        self.assertTrue(audit(self.record,self.root)[0])
    def test_unauthorized_ultra(self):
        self.record['tasks'][0]['mode_evidence'].update(
            evidence_kind='host_metadata', effective_mode=None, observed_mode='ultra')
        self.assertTrue(audit(self.record,self.root)[0])
    def test_current_run_authorized_alternative(self):
        self.record['mode_policy']={
            'requested_mode':'pro','authorized_observed_modes':['pro','ultra'],
            'authorization_scope':'this_run','authorization_receipt':'authorization.json'}
        self.record['tasks'][0]['mode_evidence'].update(
            evidence_kind='host_metadata', effective_mode=None, observed_mode='ultra')
        errors,warnings=audit(self.record,self.root)
        self.assertFalse(errors);self.assertTrue(any('do not call this Pro' in w for w in warnings))
    def test_previous_run_authorization_rejected(self):
        self.record['mode_policy']={
            'requested_mode':'pro','authorized_observed_modes':['pro','ultra'],
            'authorization_scope':'previous_run','authorization_receipt':'authorization.json'}
        self.assertTrue(audit(self.record,self.root)[0])
    def test_alternative_needs_authorization_artifact(self):
        self.record['mode_policy']={
            'requested_mode':'pro','authorized_observed_modes':['pro','ultra'],
            'authorization_scope':'this_run','authorization_receipt':'missing.md'}
        self.assertTrue(audit(self.record,self.root)[0])
    def test_request_only_is_not_evidence(self):
        self.record['tasks'][0]['mode_evidence'].update(
            evidence_kind='requested_only', effective_mode=None, observed_mode=None)
        self.assertTrue(audit(self.record,self.root)[0])
    def test_invalid_policy(self):
        self.record['mode_policy']='ultra'
        self.assertTrue(audit(self.record,self.root)[0])
    def test_pending_template_does_not_claim_execution(self):
        record={'run_id':'pending','status':'pending','tasks':[],
                'mode_policy':{'requested_mode':'pro','authorized_observed_modes':['pro'],
                               'authorization_scope':'this_run','authorization_receipt':None},
                'mode_evidence':{'requested_mode':'pro','observed_mode':None,
                                 'effective_mode':None,'evidence_kind':'unknown'}}
        self.assertFalse(audit(record,self.root)[0])

    def alternative(self):
        self.record['mode_policy'] = {
            'requested_mode': 'pro', 'authorized_observed_modes': ['pro', 'ultra'],
            'authorization_scope': 'this_run', 'authorization_receipt': 'authorization.json'}
        self.record['tasks'][0]['mode_evidence'].update(
            evidence_kind='host_metadata', effective_mode=None, observed_mode='ultra')

    def test_legacy_receipt_api_and_fields(self):
        request = make_request('fixture-model', 'max')
        result = make_receipt({'model': 'fixture-model', 'reasoning': {'mode': 'pro', 'effort': 'max'}}, request)
        self.assertIs(make_receipt, receipt_from)
        self.assertEqual(result['model'], result['model_returned'])
        self.assertEqual(result['checked_utc'], result['checked_at_utc'])
        self.assertEqual(result['returned_effort'], 'max')

    def test_malformed_response_reasoning_is_not_pro(self):
        for reasoning in (None, [], ['pro'], 'pro', 1):
            with self.subTest(reasoning=reasoning):
                result = make_receipt({'reasoning': reasoning}, make_request('fixture-model', 'max'))
                self.assertIsNone(result['effective_mode'])
                self.assertIsNone(result['returned_effort'])
                self.assertFalse(result['pro_verified'])

    def test_ui_cannot_relabel_unknown_provider_mode_as_pro(self):
        self.record['mode_evidence'].update(evidence_kind='ui_user_attested', observed_mode='pro')
        self.assertTrue(audit(self.record, self.root)[0])

    def test_legacy_ultra_override_is_not_a_new_policy(self):
        self.record['tasks'][0]['mode_evidence'].update(
            requested_mode='user_authorized_ultra', effective_mode=None,
            evidence_kind='host_metadata_plus_user_authorization', effort='ultra')
        self.assertTrue(audit(self.record, self.root)[0])

    def test_authorization_must_be_structured(self):
        self.alternative()
        self.record['mode_policy']['authorization_receipt'] = 'a.md'
        self.assertTrue(audit(self.record, self.root)[0])

    def test_authorization_bound_to_run(self):
        self.alternative()
        data = json.loads((self.root / 'authorization.json').read_text())
        data['run_id'] = 'previous-campaign'
        (self.root / 'authorization.json').write_text(json.dumps(data))
        self.assertTrue(audit(self.record, self.root)[0])

    def test_authorization_modes_must_match(self):
        self.alternative()
        data = json.loads((self.root / 'authorization.json').read_text())
        data['authorized_observed_modes'] = ['pro']
        (self.root / 'authorization.json').write_text(json.dumps(data))
        self.assertTrue(audit(self.record, self.root)[0])

    def test_authorization_receipt_scope_must_be_current(self):
        self.alternative()
        data = json.loads((self.root / 'authorization.json').read_text())
        data['authorization_scope'] = 'previous_run'
        (self.root / 'authorization.json').write_text(json.dumps(data))
        self.assertTrue(audit(self.record, self.root)[0])

    def test_authorization_requested_mode_must_match(self):
        self.alternative()
        data = json.loads((self.root / 'authorization.json').read_text())
        data['requested_mode'] = 'standard'
        (self.root / 'authorization.json').write_text(json.dumps(data))
        self.assertTrue(audit(self.record, self.root)[0])

    def test_malformed_dependencies_report_errors(self):
        for deps in ([{}], [[]], [1], [''], {}, None):
            with self.subTest(deps=deps):
                record = copy.deepcopy(self.record)
                record['tasks'][0]['depends_on'] = deps
                self.assertTrue(audit(record, self.root)[0])

    def test_malformed_record_and_task_fields_report_errors(self):
        for field, value in [('status', []), ('task_id', {}), ('isolation_kind', [])]:
            with self.subTest(field=field):
                record = copy.deepcopy(self.record)
                record['tasks'][0][field] = value
                self.assertTrue(audit(record, self.root)[0])
        for value in ([], {}, None):
            record = copy.deepcopy(self.record)
            record['status'] = value
            self.assertTrue(audit(record, self.root)[0])
        self.assertTrue(audit([], self.root)[0])

    def test_malformed_mode_and_policy_fields_report_errors(self):
        for field in ('evidence_kind', 'model', 'requested_mode', 'effective_mode'):
            with self.subTest(field=field):
                record = copy.deepcopy(self.record)
                record['mode_evidence'][field] = []
                self.assertTrue(audit(record, self.root)[0])
        for field in ('requested_mode', 'authorized_observed_modes'):
            record = copy.deepcopy(self.record)
            record['mode_policy'] = {field: {}}
            self.assertTrue(audit(record, self.root)[0])

    def test_host_receipt_context_model_and_effort_match(self):
        self.alternative()
        mode = self.record['tasks'][0]['mode_evidence']
        mode.update(receipt_path='host.json', context_id='context-a', effort='ultra')
        data = [{'context_id': 'context-a', 'host_selected': {'model': mode['model'], 'effort': 'ultra'}}]
        path = self.root / 'host.json'
        path.write_text(json.dumps(data))
        self.assertFalse(audit(self.record, self.root)[0])
        data[0]['host_selected']['model'] = 'another-model'
        path.write_text(json.dumps(data))
        self.assertTrue(audit(self.record, self.root)[0])
        data[0]['host_selected']['model'] = mode['model']
        data[0]['host_selected']['effort'] = 'high'
        path.write_text(json.dumps(data))
        self.assertTrue(audit(self.record, self.root)[0])
        mode['context_id'] = 'another-context'
        self.assertTrue(audit(self.record, self.root)[0])

    def test_structured_receipt_cannot_contradict_declared_mode(self):
        mode = self.record['mode_evidence']
        mode['receipt_path'] = 'host.json'
        (self.root / 'host.json').write_text(json.dumps({'model': mode['model'], 'effective_mode': 'standard'}))
        self.assertTrue(audit(self.record, self.root)[0])

    def test_another_workers_valid_receipt_is_rejected(self):
        self.alternative()
        mode = self.record['tasks'][0]['mode_evidence']
        mode.update(receipt_path='host.json', context_id='another-worker')
        (self.root / 'host.json').write_text(json.dumps([
            {'context_id': 'another-worker', 'host_selected': {'model': mode['model']}}]))
        self.assertTrue(audit(self.record, self.root)[0])

    def test_object_receipt_context_must_match_owning_task(self):
        self.alternative()
        mode = self.record['tasks'][0]['mode_evidence']
        mode.update(receipt_path='host.json', context_id='context-a')
        (self.root / 'host.json').write_text(json.dumps({
            'context_id': 'another-worker', 'host_selected': {'model': mode['model']}}))
        self.assertTrue(audit(self.record, self.root)[0])

    def test_receipt_symlink_cannot_escape_run(self):
        with tempfile.TemporaryDirectory() as outside:
            path = Path(outside) / 'receipt.json'
            path.write_text('{}')
            (self.root / 'linked.json').symlink_to(path)
            self.record['mode_evidence']['receipt_path'] = 'linked.json'
            self.assertTrue(audit(self.record, self.root)[0])

    def test_held_out_claim_requires_runtime_isolation(self):
        self.record['probes'] = [{
            'probe_id': 'p', 'executed': True, 'command': 'fixture-only',
            'contract_path': 'a.md', 'code_path': 'a.md', 'config_path': 'a.md',
            'log_path': 'a.md', 'result_path': 'a.md',
            'claims_held_out': True, 'evaluation_isolation': 'instruction_only'}]
        self.assertTrue(audit(self.record, self.root)[0])

    def test_duplicate_task_ids_rejected(self):
        self.record['tasks'].append(copy.deepcopy(self.record['tasks'][0]))
        self.assertTrue(audit(self.record, self.root)[0])


if __name__ == '__main__':
    unittest.main()
