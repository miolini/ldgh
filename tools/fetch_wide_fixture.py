"""Fetch an unmodified wide MLP slice through the local range route."""
from concurrent.futures import ThreadPoolExecutor
import hashlib, json, struct, urllib.request
import numpy as np
from fetch_mirai_fixtures import ROOT, REVISION
from mirai_reference import bf16, reference
URL = 'http://localhost:8000/models/qwen38/vllm/trellis.mirai'
SIZE = 8449310484

def read_range(start, length):
    parts=[]
    for offset in range(0,length,128*1024):
        count=min(128*1024,length-offset); first=start+offset; last=first+count-1
        request=urllib.request.Request(URL,headers={'Range':f'bytes={first}-{last}'})
        with urllib.request.urlopen(request,timeout=90) as response:
            if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {first}-{last}/{SIZE}': raise RuntimeError('Local range mismatch')
            data=response.read(count+1)
            if len(data)!=count: raise RuntimeError('Incomplete local range')
            parts.append(data)
    return b''.join(parts)

header=json.loads((ROOT/'model-source/mirai-sidecar-header.json').read_text())
base=8+struct.unpack('<Q',read_range(0,8))[0]
name='layers.0.mlp.down_proj'; rows=96; columns=17408; packets=272
directory=ROOT/'webgpu-fixtures/mlp-down';directory.mkdir(exist_ok=True)
roles={'packets':(name+'.0.packets',rows*packets*16),'entries':(name+'.0.entries',rows*packets*2),'rowscale':(name+'.0.rowscale',rows*4),'codebook':('codebook.v2',None),'signs':('rotation.signs_17408',None),'small_q':('rotation.q_17408',None)}
def fetch(item):
    role,(key,length)=item; info=header[key]; first,last=info['data_offsets']; data=read_range(base+first,length if length is not None else last-first)
    filename=role+'.bin'; (directory/filename).write_bytes(data)
    return role,{'file':filename,'dtype':info['dtype'],'source_tensor':key,'source_shape':info['shape'],'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
with ThreadPoolExecutor(max_workers=4) as pool: tensors=dict(pool.map(fetch,roles.items()))
manifest={'source_repository':'trymirai/Qwen3.8-27B-S-experimental','source_revision':REVISION,'source_file':'vllm/trellis.mirai','layer':name,'format':'v2t4','rows':rows,'columns':columns,'packets_per_row':packets,'rotation_order':17,'tensors':tensors}
(directory/'fixture.json').write_text(json.dumps(manifest,indent=2)+'\n')
indices=np.arange(columns,dtype=np.float32); x=bf16(np.sin(indices*np.float32(.017))*np.float32(.1)+np.cos(indices*np.float32(.031))*np.float32(.05))
expected,rotated,coarse,fine=reference(directory,manifest,x)
x.tofile(directory/'input.bin');expected.tofile(directory/'expected.bin');rotated.tofile(directory/'rotated.bin');np.stack([coarse,fine],axis=1).astype('<i4').tofile(directory/'planes.bin')
index=ROOT/'webgpu-fixtures/index.json';fixtures=json.loads(index.read_text())
if 'mlp-down' not in fixtures: fixtures.append('mlp-down');index.write_text(json.dumps(fixtures)+'\n')
print(json.dumps({'sourceBytes':sum(t['bytes'] for t in tensors.values()),'columns':columns,'rotationOrder':17,'rows':rows}))
