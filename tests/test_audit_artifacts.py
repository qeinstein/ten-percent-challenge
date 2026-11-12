import gzip,json,subprocess,sys,tempfile,unittest
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'gpu'))
from weightspace import compile_matrix

class ArtifactAuditTests(unittest.TestCase):
    def test_reports_changed_codes_and_invalid_scale(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); first=root/'a';second=root/'b';first.mkdir();second.mkdir()
            w=torch.arange(1,49,dtype=torch.float32).reshape(3,16)
            x=compile_matrix(w,torch.ones(16),density=.25,group_size=8,quantization='int4')
            y=compile_matrix(w,torch.ones(16),density=.5,group_size=8,quantization='int4');y['scales'][0,0]=-1
            for path,obj in [(first,x),(second,y)]:
                with gzip.open(path/'model_layers_0_test.pt.gz','wb') as h:torch.save(obj,h)
            output=root/'report.json'
            subprocess.run([sys.executable,str(ROOT/'gpu/audit_artifacts.py'),str(first),str(second),'--output',str(output)],check=True,capture_output=True)
            result=json.loads(output.read_text())
            self.assertEqual(result['operators'],1);self.assertEqual(result['same_codes'],0);self.assertEqual(result['with_errors'],1)
            self.assertIn('negative_scale',result['details'][0]['second_errors'])

if __name__=='__main__':unittest.main()
