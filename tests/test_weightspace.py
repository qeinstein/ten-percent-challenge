import io,sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'gpu'))
from weightspace import compile_matrix,reconstruct_matrix,add_task_weighted_residual
from my_tools import make,TOOLS

class WeightSpaceTests(unittest.TestCase):
    def test_round_trip_shape_support_and_activation_saliency(self):
        w=torch.arange(1,33,dtype=torch.float32).reshape(2,16)
        importance=torch.tensor([100.]+[1.]*15)
        packed=compile_matrix(w,importance,density=.25,group_size=8,rows_per_chunk=1)
        restored=reconstruct_matrix(packed,rows_per_chunk=1)
        self.assertEqual(tuple(restored.shape),tuple(w.shape))
        self.assertEqual(int(torch.count_nonzero(restored)),8)
        self.assertTrue(torch.all(restored[:,0]!=0))
        self.assertEqual(packed['density_per_group'],.25)
    def test_odd_dimensions_round_trip(self):
        w=torch.randn(3,13); a=compile_matrix(w,torch.ones(13),.5,group_size=8,rows_per_chunk=2)
        self.assertEqual(tuple(reconstruct_matrix(a,rows_per_chunk=2).shape),tuple(w.shape))
        self.assertAlmostEqual(a['density_actual'],round(.5*13)/13)
    def test_task_importance_reallocates_without_changing_parameter_budget(self):
        w=torch.arange(1,65,dtype=torch.float32).reshape(2,32)
        a=compile_matrix(w,torch.ones(32),density=.25,group_size=8,output_importance=torch.tensor([100.,1.]))
        decoded=reconstruct_matrix(a)
        per_row=torch.count_nonzero(decoded,dim=1)
        self.assertEqual(int(per_row.sum()),16)
        self.assertGreater(int(per_row[0]),int(per_row[1]))
    def test_variable_row_budget_keeps_each_rows_most_salient_weights(self):
        w=torch.arange(1,65,dtype=torch.float32).reshape(2,32)
        a=compile_matrix(w,torch.ones(32),density=.25,group_size=8,output_importance=torch.tensor([100.,1.]))
        decoded=reconstruct_matrix(a)
        for original,row in zip(w,decoded):
            kept=torch.nonzero(row,as_tuple=False).flatten()
            expected=original.abs().topk(len(kept)).indices.sort().values
            self.assertTrue(torch.equal(kept,expected))
    def test_int4_quantizer_decodes_signed_levels_and_beats_ternary_error(self):
        w=torch.tensor([[.1,.2,.35,.6,1.,-1.5,-2.5,-4.]])
        act=torch.ones(8)
        tri=reconstruct_matrix(compile_matrix(w,act,density=1.,group_size=8,quantization='ternary'))
        q4=reconstruct_matrix(compile_matrix(w,act,density=1.,group_size=8,quantization='int4'))
        self.assertLess(float((q4-w).square().mean()),float((tri-w).square().mean()))
    def test_task_weighted_residual_reduces_weighted_error_and_roundtrips(self):
        torch.manual_seed(31)
        w=torch.randn(24,4)@torch.randn(4,32); x=torch.linspace(.5,2.,32); y=torch.linspace(.5,2.,24)
        a=compile_matrix(w,x,density=.125,group_size=8,quantization='int4')
        base=reconstruct_matrix(a); a=add_task_weighted_residual(a,w,x,y,rank=4)
        restored=reconstruct_matrix(a,rows_per_chunk=5)
        error=lambda z: ((w-z)*y[:,None]*x[None,:]).square().sum()
        self.assertLess(float(error(restored)),float(error(base)))
        self.assertEqual(a['correction_left'].dtype,torch.float16)
        self.assertEqual(tuple(a['correction_right'].shape),(4,32))
        data=io.BytesIO();torch.save(a,data);data.seek(0)
        self.assertTrue(torch.equal(restored,reconstruct_matrix(torch.load(data,weights_only=True),rows_per_chunk=5)))
    def test_data_separates_wording_and_correct_options(self):
        train=make('calib',256,1); dev=make('dev',256,2); held=make('heldout',256,3)
        self.assertFalse(set(r['text'] for r in train)&set(r['text'] for r in held))
        self.assertFalse(set(r['text'] for r in dev)&set(r['text'] for r in held))
        self.assertTrue(all(0<=r['answer']<5 and len(set(r['names']))==5 for r in train+dev+held))
        self.assertEqual(train,make('calib',256,1))
        self.assertLess(abs(sum(r['answer']==4 for r in train)/len(train)-.2),.01)

if __name__=='__main__': unittest.main()

class FeedbackTests(unittest.TestCase):
    def test_feedback_adjusts_named_tool_independent_of_position_and_roundtrips(self):
        import tempfile
        from feedback import Feedback
        f=Feedback(strength=1); h=torch.tensor([1.,0.,0.]); f.ingest(h,'send_mail: sends email')
        options=['other: a','send_mail: sends email','None of these tools can do this']
        self.assertEqual(int(f.adjust(torch.zeros(3),h,options).argmax()),1)
        shuffled=[options[1],options[0],options[2]]
        self.assertEqual(int(f.adjust(torch.zeros(3),h,shuffled).argmax()),0)
        with tempfile.NamedTemporaryFile() as temp:
            f.save(temp.name); loaded=Feedback.load(temp.name)
            self.assertTrue(torch.equal(f.adjust(torch.zeros(3),h,options),loaded.adjust(torch.zeros(3),h,options)))
    def test_feedback_without_annotations_preserves_logits(self):
        from feedback import Feedback
        logits=torch.tensor([2.,1.]); self.assertTrue(torch.equal(Feedback().adjust(logits,torch.ones(3),['a','b']),logits))
