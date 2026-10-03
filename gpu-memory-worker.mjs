self.onmessage = async () => {
  let device; const buffers=[]; let allocated=0, lastName='';
  try {
    const plan = await fetch('./gpu-memory-plan.json').then(r=>r.json());
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('WebGPU adapter unavailable');
    device = await adapter.requestDevice({requiredLimits:{maxBufferSize:512*1024*1024,maxStorageBufferBindingSize:512*1024*1024,maxComputeWorkgroupStorageSize:17408}});
    let loss; device.lost.then(info=>{loss=info.message;});
    for (let index=0; index<plan.buffers.length; index++) {
      const item=plan.buffers[index]; lastName=item.name;
      device.pushErrorScope('out-of-memory'); device.pushErrorScope('validation');
      const buffer=device.createBuffer({label:item.name,size:item.bytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
      buffers.push(buffer);
      const encoder=device.createCommandEncoder(); encoder.clearBuffer(buffer);
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      const validation=await device.popErrorScope(), memory=await device.popErrorScope();
      if (validation||memory||loss) throw new Error((validation||memory)?.message||loss);
      allocated+=item.bytes;
      if (index%64===0) self.postMessage({text:`Allocated and cleared ${(allocated/1e9).toFixed(2)} / ${(plan.totalBytes/1e9).toFixed(2)} GB (${index+1}/${plan.buffers.length} buffers)…`});
    }
    self.postMessage({done:true,checkedAt:new Date().toISOString(),allocationPassed:true,revision:plan.revision,allocatedBytes:allocated,buffers:buffers.length,scope:plan.scope,fullModelInferenceVerified:false});
  } catch(error) {self.postMessage({done:true,checkedAt:new Date().toISOString(),allocationPassed:false,allocatedBytes:allocated,lastBuffer:lastName,error:error.message,fullModelInferenceVerified:false});}
  finally {buffers.forEach(buffer=>buffer.destroy());device?.destroy();}
};
