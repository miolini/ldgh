"""Describe buffers allocated by the full browser runtime, from pinned shapes."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
h = json.loads((ROOT / 'model-source/mirai-sidecar-header.json').read_text())
dense = json.loads((ROOT / 'model-source/vllm/dense-manifest.json').read_text())
layers = json.loads(h['__metadata__']['layers'])
plan = []
def add(name, size): plan.append({'name':name, 'bytes':max(4, (size + 3) // 4 * 4)})
def raw(name):
    a,b = h[name]['data_offsets']; return b-a
def count(name):
    result = 1
    for v in h[name]['shape']: result *= v
    return result
for name,meta in layers.items():
    n = meta['in_features']; order = h[f'rotation.q_{n}']['shape'][0]
    add(name + '.output',meta['out_features']*4)
    for i,block in enumerate(meta['blocks']):
        prefix = f'{name}.{i}.'; rows = count(prefix+'rowscale')
        for role,size in {'input':n*4, 'signs':n*4, 'small_q':order*order*4, 'rotated':n*4, 'maxima':order*4, 'planes':n*2, 'stats':32, 'packets':raw(prefix+'packets'), 'entries':count(prefix+'entries')*4, 'rowscale':rows*4, 'codebook':raw('codebook.'+block['format'][:2]), 'row_map':rows*4}.items(): add(prefix+role,size)
for kind in ('embedding','head'):
    for role in ('codes','row_scales','ladder_indices','signs'): add(kind+'.'+role,raw(kind+'.'+role))
    add(kind+'.ladder',count(kind+'.ladder')*4)
for role,size in {'table':count('embedding.table')*4,'tokenIds':4,'embedding':5120*4,'headInput':5120*4,'rotated':5120*4,'logits':248320*4}.items(): add('vocabulary.'+role,size)
for index in range(64):
    prefix=f'layer.{index}.'
    for role,size in {'inputNormOutput':5120*4,'inputNormWeights':5120*4,'postNormOutput':5120*4,'postNormWeights':5120*4,'attentionResidual':5120*4,'mlpResidual':5120*4,'activation':17408*4}.items(): add(prefix+role,size)
    if (index+1)%4==0:
        sizes={'normWeights':512*4,'query':6144*4,'gate':6144*4,'keyCache':4096*1024*4,'valueCache':4096*1024*4,'parameters':4,'cosSin':64*4,'output':6144*4,'roundedSigmoid':65536*4}
    else:
        sizes={'conv':10240*4*4,'decay':96*4,'history':10240*3*4,'qkv':10240*4,'normalized':4096*4,'state':48*128*128*4,'recurrent':6144*4,'normOutput':6144*4,'normWeight':128*4}
    for role,size in sizes.items(): add(prefix+'attention.'+role,size)
add('finalNorm.output',5120*4);add('finalNorm.weight',5120*4)
result={'revision':dense['revision'],'contextCapacity':4096,'scope':'Exact full-runtime buffer shapes; allocation only, without model inference or pipeline overhead','buffers':plan,'totalBytes':sum(x['bytes'] for x in plan)}
(ROOT/'gpu-memory-plan.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'buffers':len(plan),'totalBytes':result['totalBytes'],'GiB':result['totalBytes']/2**30}))
