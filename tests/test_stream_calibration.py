import math,sys,unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'gpu'))
import qwen_nec
class Block(nn.Module):
    def __init__(self): super().__init__(); self.proj=nn.Linear(2,2,bias=False)
    def forward(self,x): return self.proj(x)
class Core(nn.Module):
    def __init__(self): super().__init__(); self.layers=nn.ModuleList([Block()])
    def forward(self,input_ids,attention_mask,use_cache):
        x=input_ids.float(); return self.layers[0](torch.stack((x,2*x),-1))
class Model(nn.Module):
    def __init__(self): super().__init__(); self.model=Core()
class Tokens(dict):
    __getattr__=dict.__getitem__
class Tokenizer:
    def __call__(self,prompts,**kwargs):
        return Tokens(input_ids=torch.tensor([[0,1,2,3],[1,2,3,4]]),attention_mask=torch.tensor([[0,1,1,1],[1,1,1,1]]))
class StreamTests(unittest.TestCase):
    def test_padded_batches_accumulate_only_selected_real_positions(self):
        qwen_nec.DEVICE='cpu'; rows=[{'inst':'x','text':'x','names':['x']},{'inst':'y','text':'y','names':['y']}]
        actual=qwen_nec.capture(Model(),Tokenizer(),rows,torch.arange(5),2,512,batch_size=2)['model.layers.0.proj']
        expected=torch.tensor([math.sqrt(6.75),2*math.sqrt(6.75)])
        torch.testing.assert_close(actual,expected)
if __name__=='__main__':unittest.main()
