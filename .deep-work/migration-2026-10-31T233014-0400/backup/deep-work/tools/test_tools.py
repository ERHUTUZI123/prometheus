"""Offline tests only: no paid calls or scientific experiments."""
import copy
from pathlib import Path
import tempfile
import unittest
from audit import audit
from pro_probe import make_receipt, make_request

class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for name in ['receipt.json','inputs.json','a.md','b.md','result.md']:
            (self.root/name).write_text('fixture')
        self.mode = {'model':'gpt-6-astra','requested_mode':'pro','effective_mode':'pro',
                     'evidence_kind':'provider_metadata','receipt_path':'receipt.json'}
        self.record = {'run_id':'test','status':'completed','mode_evidence':copy.deepcopy(self.mode),
                       'tasks':[{'task_id':'a','status':'completed','mode_evidence':copy.deepcopy(self.mode),
                                 'output_path':'a.md','input_manifest':'inputs.json','independent_branch':True,
                                 'context_id':'A','prior_report_seen_before_freeze':False,
                                 'isolation_kind':'runtime_enforced'}], 'probes':[], 'final_report':'result.md'}
    def tearDown(self): self.tmp.cleanup()
    def test_valid(self): self.assertFalse(audit(self.record,self.root,True)[0])
    def test_high_not_pro(self):
        self.record['tasks'][0]['mode_evidence']['effective_mode']='standard'
        self.assertTrue(audit(self.record,self.root)[0])
    def test_unknown_mode(self):
        self.record['mode_evidence']['effective_mode']=None
        self.assertTrue(audit(self.record,self.root)[0])
    def test_duplicate_context(self):
        t=copy.deepcopy(self.record['tasks'][0]);t['task_id']='b';t['output_path']='b.md'
        self.record['tasks'].append(t)
        self.assertTrue(audit(self.record,self.root)[0])
    def test_missing_output(self):
        self.record['tasks'][0]['output_path']='absent.md'
        self.assertTrue(audit(self.record,self.root)[0])
    def test_escape(self):
        self.record['tasks'][0]['output_path']='../result.md'
        self.assertTrue(audit(self.record,self.root)[0])
    def test_cycle(self):
        self.record['tasks'][0]['depends_on']=['a']
        self.assertTrue(audit(self.record,self.root)[0])
    def test_ui_evidence(self):
        self.record['tasks'][0]['mode_evidence']['evidence_kind']='ui_user_attested'
        errors,warnings=audit(self.record,self.root)
        self.assertFalse(errors);self.assertTrue(any('attestation' in w for w in warnings))
    def test_prior_report(self):
        self.record['tasks'][0]['prior_report_seen_before_freeze']=True
        self.assertTrue(audit(self.record,self.root)[0])
    def test_request(self):
        self.assertEqual(make_request('gpt-6-astra','max')['reasoning'],{'mode':'pro','effort':'max'})
    def test_request_does_not_prove_mode(self):
        r=make_receipt({'model':'gpt-6-astra'},make_request('gpt-6-astra','max'))
        self.assertFalse(r['pro_verified']);self.assertEqual(r['evidence_kind'],'requested_only')
    def test_returned_mode(self):
        r=make_receipt({'model':'gpt-6-astra','reasoning':{'mode':'pro'}},make_request('gpt-6-astra','max'))
        self.assertTrue(r['pro_verified'])

if __name__=='__main__': unittest.main()
