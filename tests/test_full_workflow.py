import contextlib,hashlib,io,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'gpu'))
import qwen_nec
from transformers import Qwen2Config,Qwen2ForCausalLM
class Enc(dict): __getattr__=dict.__getitem__
class Tok:
    pad_token=None; eos_token='<eos>'; unk_token_id=0; eos_token_id=0; all_special_ids=[0]; padding_side='left'
    def __len__(self):return 32
    def encode(self,text,add_special_tokens=False):return [ord(text[-1])-64] if text[-1] in 'ABCDE' else [1,2,3,4]
    def __call__(self,text,**kw):
        n=len(text) if isinstance(text,list) else 1
        return Enc(input_ids=torch.tensor([[1,2,3,4]]*n),attention_mask=torch.ones(n,4,dtype=torch.long))
    def save_pretrained(self,path):
        path=Path(path);path.mkdir(parents=True,exist_ok=True);(path/'tokenizer.json').write_text('{}')
class FullWorkflow(unittest.TestCase):
    def test_full_tiny_qwen_pipeline_and_feedback(self):
        torch.set_num_threads(2)
        cfg=Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=32)
        model=Qwen2ForCausalLM(cfg).eval()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data=root/'data';data.mkdir(); files={}
            for split,n in [('calib',8),('dev',4),('feedback',3)]:
                rows=[{'id':split+str(i),'inst':'choose','text':'request'+str(i),'names':['first: a','second: b','third: c','fourth: d','None of these tools can do this'],'answer':0} for i in range(n)]
                payload=''.join(json.dumps(r)+'\n' for r in rows).encode();(data/(split+'.jsonl')).write_bytes(payload);files[split+'.jsonl']={'sha256':hashlib.sha256(payload).hexdigest()}
            (data/'manifest.json').write_text(json.dumps({'files':files}))
            args=['qwen_nec.py','--device','cpu','--model','mock','--revision','mock','--data-dir',str(data),'--calibration','8','--calibration-positions','8','--dev','4','--output',str(root/'out'),'--limit-seconds','120']
            with patch.object(sys,'argv',args),patch.object(qwen_nec.AutoTokenizer,'from_pretrained',return_value=Tok()),patch.object(qwen_nec.AutoModelForCausalLM,'from_pretrained',return_value=model),contextlib.redirect_stdout(io.StringIO()): qwen_nec.main()
            report=json.loads((root/'out/report.json').read_text())
            self.assertEqual(report['operators'],14);self.assertEqual(report['feedback']['annotations'],1);self.assertEqual(report['feedback']['unseen_new_after']['total'],1)
            self.assertTrue((root/'out/package/feedback.pt').exists());self.assertEqual(json.loads((root/'out/status.json').read_text())['status'],'complete')
if __name__=='__main__':unittest.main()
