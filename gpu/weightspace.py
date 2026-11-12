"""Task-activation-aware sparse ternary weight-space compiler.

Zero gradient steps. Per group, retain the weights with the largest estimated
activation-weighted impact and solve a weighted least-squares ternary scale.
The support is implicit in the packed 2-bit values and entropy-coded on disk.
"""
import torch

@torch.no_grad()
def compile_matrix(weight, activation_rms, density=.05, group_size=128, rows_per_chunk=1024,
                   allocation='rowwise', output_importance=None, quantization='ternary'):
    if not 0 < density <= 1 or group_size < 4 or group_size % 4: raise ValueError('invalid density or group size')
    if allocation not in ('rowwise','groupwise'): raise ValueError('allocation must be rowwise or groupwise')
    if quantization not in ('ternary','int4'): raise ValueError('quantization must be ternary or int4')
    rows,cols=weight.shape; groups=(cols+group_size-1)//group_size; padded=groups*group_size
    imp=torch.nn.functional.pad(activation_rms.float(),(0,padded-cols)).reshape(groups,group_size)
    k=max(1,min(group_size,round(density*group_size))); row_k=max(1,min(cols,round(density*cols))); packed=[]; scales=[]
    if output_importance is not None:
        if output_importance.numel()!=rows: raise ValueError('output importance must have one value per output row')
        # Spend the same total nonzero budget per matrix while giving salient
        # output channels more support. A narrow power and 4x cap limit noise.
        g=output_importance.detach().float().clamp_min(1e-20)
        relative=(g/g.median().clamp_min(1e-20)).pow(.25).clamp(.5,2.)
        lo,hi=0.,max(2.,row_k*4.)
        for _ in range(24):
            mid=(lo+hi)/2
            if (relative*mid).clamp(1,cols).mean()>row_k: hi=mid
            else: lo=mid
        row_counts=(relative*((lo+hi)/2)).clamp(1,cols).round().long()
        # Correct integer rounding so this layer stays on its requested budget.
        delta=rows*row_k-int(row_counts.sum())
        while delta:
            eligible=row_counts<cols if delta>0 else row_counts>1
            order=torch.argsort(g[eligible],descending=(delta>0))
            if not order.numel(): raise RuntimeError('could not balance per-row keep budget')
            step=1 if delta>0 else -1
            take=min(abs(delta),int(order.numel()))
            indices=torch.nonzero(eligible,as_tuple=False).flatten()[order[:take]]
            row_counts[indices]+=step
            delta-=step*take
    else:
        row_counts=None
    for start in range(0,rows,rows_per_chunk):
        chunk=weight[start:start+rows_per_chunk].float(); chunk_rows=chunk.shape[0]
        w=torch.nn.functional.pad(chunk,(0,padded-cols)).reshape(-1,groups,group_size)
        # WANDA-style per-output ranking: do not force the same number of
        # survivors into every input block, which can discard salient channels.
        score=w.abs()*imp.unsqueeze(0)
        if allocation=='rowwise':
            flat=score.reshape(-1,padded)
            if row_counts is None:
                indices=flat.topk(row_k,dim=-1,sorted=False).indices
                keep_flat=torch.zeros_like(flat,dtype=torch.bool).scatter_(1,indices,True)
            else:
                counts=row_counts[start:start+chunk_rows].to(flat.device)
                max_k=int(counts.max())
                # Rows have different budgets, so the first `counts` entries
                # must be the most salient entries in descending order.
                indices=flat.topk(max_k,dim=-1,sorted=True).indices
                ranks=torch.arange(max_k,device=flat.device).unsqueeze(0)<counts.unsqueeze(1)
                keep_flat=torch.zeros_like(flat,dtype=torch.bool).scatter_(1,indices,ranks)
            keep=keep_flat.reshape(-1,groups,group_size)
        else:
            indices=score.topk(k,dim=-1,sorted=False).indices
            keep=torch.zeros_like(w,dtype=torch.bool).scatter_(-1,indices,True)
        weights=imp.square().unsqueeze(0); denominator=(keep*weights).sum(-1).clamp_min(1e-20)
        if quantization=='ternary':
            scale=((w.abs()*keep*weights).sum(-1)/denominator).half()
            code=torch.where(keep,torch.where(w>=0,1,2),0).to(torch.uint8).reshape(-1)
            code=code[0::4] | (code[1::4]<<2) | (code[2::4]<<4) | (code[3::4]<<6)
        else:
            # Four-bit signed levels improve magnitude fidelity while sparse
            # zero codes remain highly compressible under gzip.
            scale=((w.abs()*keep*weights).sum(-1)/denominator/3.5).clamp_min(1e-10)
            for _ in range(3):
                magnitude=(w.abs()/scale.unsqueeze(-1).clamp_min(1e-10)).round().clamp(1,7)
                qden=(magnitude.square()*keep*weights).sum(-1).clamp_min(1e-20)
                scale=((w.abs()*magnitude*keep*weights).sum(-1)/qden).clamp_min(1e-10)
            magnitude=(w.abs()/scale.unsqueeze(-1)).round().clamp(1,7).to(torch.int8)
            signed=torch.where(w>=0,magnitude,-magnitude)
            code=torch.where(keep,(signed.to(torch.int16)+8).to(torch.uint8),torch.zeros_like(signed,dtype=torch.uint8)).reshape(-1)
            code=code[0::2] | (code[1::2]<<4)
            scale=scale.half()
        packed.append(code.cpu()); scales.append(scale.cpu())
    nominal_density=(row_k/cols if allocation=='rowwise' else k/group_size)
    return {'kind':'task_sparse_int4_v1' if quantization=='int4' else 'task_sparse_ternary_v2','shape':(rows,cols),'group_size':group_size,
            'allocation':allocation,'quantization':quantization,'density_per_group':nominal_density,
            'density_requested':float(density),'density_actual':nominal_density,
            'density_note':'nominal base support; padding, zeros or residual corrections change decoded nonzeros',
            'codes':torch.cat(packed),'scales':torch.cat(scales)}

@torch.no_grad()
def reconstruct_matrix(artifact, device='cpu', rows_per_chunk=256):
    rows,cols=artifact['shape']; group=artifact['group_size']; groups=(cols+group-1)//group; padded=groups*group
    packed=artifact['codes']; scale=artifact['scales']; out=torch.empty((rows,cols),device=device,dtype=torch.float32); row_bytes=padded//4
    if artifact.get('kind')=='task_sparse_int4_v1':
        row_bytes=padded//2
    correction_right=artifact.get('correction_right')
    if correction_right is not None: correction_right=correction_right.to(device).float()
    for start in range(0,rows,rows_per_chunk):
        end=min(rows,start+rows_per_chunk); data=packed[start*row_bytes:end*row_bytes].to(device)
        if artifact.get('kind')=='task_sparse_int4_v1':
            codes=torch.empty(data.numel()*2,device=device,dtype=torch.uint8)
            codes[0::2]=data&15; codes[1::2]=data>>4
            codes=codes.reshape(end-start,groups,group).to(torch.int16)
            signed=torch.where(codes==0,0,codes-8).float()
        else:
            codes=torch.empty(data.numel()*4,device=device,dtype=torch.uint8)
            for shift in range(4): codes[shift::4]=(data>>(2*shift))&3
            codes=codes.reshape(end-start,groups,group)
            signed=(codes==1).float()-(codes==2).float()
        values=signed*scale[start:end].to(device).float().unsqueeze(-1)
        out[start:end]=values.reshape(end-start,padded)[:,:cols]
        if correction_right is not None:
            left=artifact['correction_left'][start:end].to(device).float()
            out[start:end].add_(left@correction_right)
    return out

@torch.no_grad()
def add_task_weighted_residual(artifact, weight, activation_rms, output_importance=None, rank=16, oversample=8):
    """Add a closed-form low-rank residual fitted in task-weighted coordinates.

    Factors are stored in the artifact and counted in its serialized bytes.
    The weighting is a compression proxy; candidate selection uses task labels.
    """
    rows,cols=weight.shape
    if rank<=0 or rank>min(rows,cols): raise ValueError('invalid residual rank')
    device=weight.device; w=weight.detach().float()
    x=activation_rms.detach().to(device).float().clamp_min(1e-20)
    x=(x/x.median().clamp_min(1e-20)).clamp(.25,4.)
    if output_importance is None: y=torch.ones(rows,device=device)
    else:
        y=output_importance.detach().to(device).float().clamp_min(1e-20)
        if y.numel()!=rows: raise ValueError('output importance shape mismatch')
        y=(y/y.median().clamp_min(1e-20)).clamp(.25,4.)
    base=reconstruct_matrix(artifact,device=device)
    residual=(w-base)*y[:,None]*x[None,:]
    q=min(min(rows,cols),rank+max(0,oversample))
    u,s,v=torch.svd_lowrank(residual,q=q,niter=2)
    left=(u[:,:rank]*s[:rank])/y[:,None]
    right=v[:,:rank].T/x[None,:]
    ln=left.norm(dim=0).clamp_min(1e-20); rn=right.norm(dim=1).clamp_min(1e-20)
    balance=(rn/ln).sqrt()
    left=left*balance; right=right/balance[:,None]
    if not torch.isfinite(left).all() or not torch.isfinite(right).all(): raise ValueError('non-finite correction factors')
    left16=left.half(); right16=right.half()
    if not torch.isfinite(left16).all() or not torch.isfinite(right16).all(): raise ValueError('correction overflows FP16 storage')
    artifact['correction_left']=left16.cpu(); artifact['correction_right']=right16.cpu()
    artifact['correction_rank']=rank
    return artifact
