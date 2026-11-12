import sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'gpu'))
import qwen_nec
from transformers import Qwen2Config,Qwen2ForCausalLM
from weightspace import compile_matrix,reconstruct_matrix
class Enc(dict):
    __getattr__=dict.__getitem__
class Tok:
    def __call__(self,text,**kw):
        count=len(text) if isinstance(text,list) else 1
        return Enc(input_ids=torch.tensor([[1,2,3,4]]*count),attention_mask=torch.ones(count,4,dtype=torch.long))
class QwenSmoke(unittest.TestCase):
    def test_qwen_capture_and_compression(self):
        torch.set_num_threads(2);qwen_nec.DEVICE='cpu'
        cfg=Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=32)
        model=Qwen2ForCausalLM(cfg).eval(); rows=[{'inst':'x','text':'x','names':['a','b','c','d','e'],'answer':0} for _ in range(4)]
        rms=qwen_nec.capture(model,Tok(),rows,torch.arange(32),2,32,batch_size=2)
        self.assertEqual(len(rms),14); name='model.layers.0.self_attn.q_proj'; self.assertEqual(tuple(rms[name].shape),(16,));self.assertTrue(torch.isfinite(rms[name]).all())
        first=qwen_nec.capture(model,Tok(),rows,torch.arange(32),2,32,batch_size=2,name_prefix='model.layers.0.')
        self.assertEqual(len(first),7); self.assertTrue(all(n.startswith('model.layers.0.') for n in first))
        later=qwen_nec.capture(model,Tok(),rows,torch.arange(32),2,32,batch_size=2,name_prefix='model.layers.1.')
        with torch.no_grad(): model.model.layers[0].mlp.down_proj.weight.zero_()
        later_changed=qwen_nec.capture(model,Tok(),rows,torch.arange(32),2,32,batch_size=2,name_prefix='model.layers.1.')
        self.assertFalse(torch.allclose(later['model.layers.1.self_attn.q_proj'],later_changed['model.layers.1.self_attn.q_proj']))
        w=dict(model.named_modules())[name].weight.detach(); artifact=compile_matrix(w,rms[name],density=.25,group_size=8)
        self.assertEqual(tuple(reconstruct_matrix(artifact).shape),tuple(w.shape))
        head=model.lm_head.weight.detach()[:5].float()
        task_rms,task_output=qwen_nec.capture_task_gradient(model,Tok(),rows,torch.arange(32),head,32,batch_size=2)
        self.assertEqual(tuple(task_output[name].shape),(16,));self.assertTrue(torch.isfinite(task_output[name]).all())
        allocated=compile_matrix(w,task_rms[name],density=.25,group_size=8,output_importance=task_output[name])
        self.assertEqual(tuple(reconstruct_matrix(allocated).shape),tuple(w.shape))
        cached,stats=qwen_nec.feature_cache(model,Tok(),rows,head,torch.arange(32),32,batch_size=2)
        diagnostic=qwen_nec.choice_diagnostics(cached)
        self.assertEqual(sum(diagnostic['prediction_counts']),len(rows));self.assertEqual(diagnostic['gold_counts'],[4,0,0,0,0])
        direct=qwen_nec.score(model,Tok(),rows,head,torch.arange(32),max_length=32,batch_size=2)
        self.assertEqual(qwen_nec.cache_accuracy(cached)['correct'],direct['correct']); self.assertEqual(stats['input_tokens'],16)
        fbrows=[dict(rows[0],id=str(i)) for i in range(3)]
        feedback,report=qwen_nec.evaluate_feedback(model,Tok(),fbrows,cached,head,torch.arange(32),32)
        self.assertEqual(report['annotations'],1); self.assertEqual(report['tune_examples'],1); self.assertEqual(report['unseen_new_after']['total'],1)
if __name__=='__main__':unittest.main()
