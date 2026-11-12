import json,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class ProtocolTests(unittest.TestCase):
    def test_failing_development_gates_never_open_exam(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); (p/'package').mkdir()
            report={'artifact_bytes':0,'artifact_ratio':.08,'baseline_dev':{'accuracy':.9},'compressed_dev':{'accuracy':.2},'feedback':{'annotations':1,'unseen_new_before':{'total':2,'correct':0},'unseen_new_after':{'correct':1},'old_dev_after':{'accuracy':.2}}}
            (p/'report.json').write_text(json.dumps(report))
            result=subprocess.run([sys.executable,str(ROOT/'gpu/final_exam.py'),'--output',str(p),'--data-dir',str(p/'does-not-exist')],cwd=ROOT,text=True,capture_output=True,check=True)
            outcome=json.loads(result.stdout); self.assertFalse(outcome['exam_used']); self.assertFalse((p/'FINAL_EXAM_CLAIMED.json').exists())
if __name__=='__main__':unittest.main()

class FinalExamTests(unittest.TestCase):
    def test_final_exam_claims_once_and_records_paired_results(self):
        import contextlib,hashlib,importlib.util,io
        from unittest.mock import patch
        spec=importlib.util.spec_from_file_location('final_exam',ROOT/'gpu/final_exam.py'); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); package=root/'package';package.mkdir();(package/'feedback.pt').write_bytes(b'x')
            data=root/'data';data.mkdir(); locked=b'{"answer":0}\n{"answer":1}\n';(data/'locked.jsonl').write_bytes(locked)
            (data/'manifest.json').write_text(json.dumps({'files':{'locked.jsonl':{'sha256':hashlib.sha256(locked).hexdigest(),'rows':2}}}))
            report={'artifact_bytes':1,'artifact_ratio':.08,'baseline_dev':{'accuracy':.5},'compressed_dev':{'accuracy':.6},'feedback':{'annotations':1,'unseen_new_before':{'total':2,'correct':0},'unseen_new_after':{'correct':1},'old_dev_after':{'accuracy':.6}}}
            (root/'report.json').write_text(json.dumps(report))
            args=['final_exam.py','--output',str(root),'--data-dir',str(data)]
            fake={'accuracy':1.,'correct':2,'total':2,'predictions':[0,1]}
            def execute(cmd,timeout):
                return {**fake,'before_feedback':dict(fake)} if '--compare-feedback' in cmd else dict(fake)
            with patch.object(sys,'argv',args),patch.object(mod,'execute',side_effect=execute),contextlib.redirect_stdout(io.StringIO()): mod.main()
            result=json.loads((root/'final_exam.json').read_text());self.assertTrue(result['passes_point_estimate']);self.assertTrue((root/'FINAL_EXAM_CLAIMED.json').exists())
            with patch.object(sys,'argv',args),patch.object(mod,'execute',side_effect=execute):
                with self.assertRaises(FileExistsError):mod.main()
