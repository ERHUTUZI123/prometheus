#!/usr/bin/env python3
"""Local offline checks; makes no API call and runs no scientific experiment."""
from __future__ import annotations
import copy
import tempfile
import unittest
from pathlib import Path
from audit import audit
from pro_probe import make_request, receipt_from


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for file in ['receipt.json', 'a.md', 'b.md', 'inputs.json', 'result.md']:
            (self.root / file).write_text('fixture', encoding='utf-8')
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
            'authorization_scope':'this_run','authorization_receipt':'receipt.json'}
        self.record['tasks'][0]['mode_evidence'].update(
            evidence_kind='host_metadata', effective_mode=None, observed_mode='ultra')
        errors,warnings=audit(self.record,self.root)
        self.assertFalse(errors);self.assertTrue(any('do not call this Pro' in w for w in warnings))
    def test_previous_run_authorization_rejected(self):
        self.record['mode_policy']={
            'requested_mode':'pro','authorized_observed_modes':['pro','ultra'],
            'authorization_scope':'previous_run','authorization_receipt':'receipt.json'}
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


if __name__ == '__main__':
    unittest.main()
