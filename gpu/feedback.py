"""Few-annotation tool feedback with frozen model, no prompt edits or retrieval.

Stores a normalized running feature mean per tool, never individual examples.
No claim of challenge acceptance or effectiveness before evaluation.
"""
import json
import torch
import torch.nn.functional as F

def tool_key(option):
    return option.split(':',1)[0].strip().casefold()

class Feedback:
    def __init__(self, strength=.25):
        self.strength=float(strength); self.means={}; self.counts={}; self._cache={}
    def ingest(self, features, correct_option):
        key=tool_key(correct_option)
        if key=='none of these tools can do this': return
        x=F.normalize(features.detach().float().reshape(-1),dim=0).cpu(); self._cache.clear()
        n=self.counts.get(key,0); self.means[key]=(self.means[key]*n+x)/(n+1) if n else x.clone(); self.counts[key]=n+1
    def adjust(self, logits, features, options):
        if not self.means: return logits
        out=logits.clone(); x=F.normalize(features.detach().float().reshape(-1),dim=0).to(logits.device)
        for i,option in enumerate(options):
            key=tool_key(option)
            if key in self.means:
                cache_key=(key,str(logits.device))
                if cache_key not in self._cache: self._cache[cache_key]=F.normalize(self.means[key].to(logits.device),dim=0)
                out[i]+=self.strength*torch.dot(x,self._cache[cache_key])
        return out
    def save(self,path):
        torch.save({'strength':self.strength,'means':self.means,'counts':self.counts},path)
    @classmethod
    def load(cls,path):
        s=torch.load(path,map_location='cpu',weights_only=True); obj=cls(s['strength']); obj.means=s['means']; obj.counts=s['counts']; return obj
