// Same-origin transport for an OpenAI-compatible provider. Never retries case actions.
export class CompletionError extends Error {
  constructor(message,{status=0,retryable=false}={}){super(message);this.name='CompletionError';this.status=status;this.retryable=retryable;}
}

function errorDetail(value){
  if(typeof value==='string')return value;
  return value?.error?.message??value?.error??value?.message??'Invalid completion response';
}

export async function readCompletionResponse(response,{onContent=()=>{},signal}={}){
  if(!response.ok){
    const detail=await response.text();
    let parsed;try{parsed=JSON.parse(detail);}catch{parsed=detail;}
    throw new CompletionError(`Local agent request failed (HTTP ${response.status}): ${String(errorDetail(parsed)).slice(0,600)}`,
      {status:response.status,retryable:[408,429,500,502,503,504].includes(response.status)});
  }
  if(!response.headers.get('content-type')?.includes('text/event-stream')){
    let payload;try{payload=await response.json();}catch{throw new CompletionError('The local agent returned incomplete JSON.',{retryable:true});}
    if(payload.error)throw new CompletionError(String(errorDetail(payload)));
    onContent(payload.choices?.[0]?.message?.content??'');return payload;
  }
  if(!response.body)throw new CompletionError('The local agent stream was empty.',{retryable:true});
  const reader=response.body.getReader(),decoder=new TextDecoder(),calls=new Map();
  let buffer='',content='',finishReason=null,usage,done=false,sawChoice=false;
  function event(frame){
    const data=frame.split('\n').filter(line=>line.startsWith('data:')).map(line=>line.slice(5).trimStart()).join('\n');
    if(!data)return;
    if(data.trim()==='[DONE]'){done=true;return;}
    let payload;try{payload=JSON.parse(data);}catch{throw new CompletionError('The local agent stream contained an incomplete response.',{retryable:true});}
    if(payload.error)throw new CompletionError(String(errorDetail(payload)),{retryable:true});
    if(payload.usage)usage=payload.usage;
    const choice=payload.choices?.find(c=>c.index===0)??payload.choices?.[0];if(!choice)return;
    sawChoice=true;
    if(choice.finish_reason)finishReason=choice.finish_reason;
    const delta=choice.delta??choice.message??{};
    if(typeof delta.content==='string'){content+=delta.content;onContent(content);}
    // Reasoning is deliberately not shown, saved, or used as the final answer.
    for(const fragment of delta.tool_calls??[]){
      const index=fragment.index??calls.size;
      const call=calls.get(index)??{id:'',type:'function',function:{name:'',arguments:''}};
      if(fragment.id)call.id+=fragment.id;
      if(fragment.type)call.type=fragment.type;
      if(fragment.function?.name)call.function.name+=fragment.function.name;
      if(fragment.function?.arguments)call.function.arguments+=fragment.function.arguments;
      calls.set(index,call);
    }
  }
  try{
    while(!done){
      if(signal?.aborted)throw signal.reason??new DOMException('Stopped','AbortError');
      const part=await reader.read();
      buffer+=decoder.decode(part.value??new Uint8Array(),{stream:!part.done});
      // Normalize CRLF only after a complete frame, so a split CR/LF is preserved.
      let match;
      while((match=/\r?\n\r?\n/.exec(buffer))){
        const frame=buffer.slice(0,match.index).replaceAll('\r\n','\n');buffer=buffer.slice(match.index+match[0].length);event(frame);
      }
      if(part.done){if(buffer.trim())event(buffer.replaceAll('\r\n','\n'));break;}
    }
    if(!sawChoice||(!done&&!finishReason))throw new CompletionError('The local agent connection ended before the reply completed.',{retryable:true});
    return {choices:[{finish_reason:finishReason,message:{role:'assistant',content:content||null,...(calls.size?{tool_calls:[...calls.entries()].sort(([a],[b])=>a-b).map(([,call])=>call)}:{})}}],usage};
  }finally{try{await reader.cancel();}catch{}reader.releaseLock();}
}

const wait=(ms,signal)=>new Promise((resolve,reject)=>{
  if(signal?.aborted){reject(signal.reason??new DOMException('Stopped','AbortError'));return;}
  const finish=()=>{signal?.removeEventListener('abort',abort);resolve();};
  const timer=setTimeout(finish,ms);
  const abort=()=>{clearTimeout(timer);reject(signal.reason??new DOMException('Stopped','AbortError'));};
  signal?.addEventListener('abort',abort,{once:true});
});
export async function requestLocalCompletion(body,{url='/api/llm/chat/completions',signal,onContent,onRetry=()=>{},fetchImpl=fetch,retries=2}={}){
  for(let attempt=0;;attempt++){
    try{
      const response=await fetchImpl(url,{method:'POST',headers:{'Content-Type':'application/json'},signal,body:JSON.stringify(body)});
      return await readCompletionResponse(response,{onContent,signal});
    }catch(error){
      if(signal?.aborted||error.name==='AbortError')throw error;
      if(attempt>=retries||!(error.retryable||error instanceof TypeError))throw error;
      onRetry(attempt+1);await wait(350*(attempt+1),signal);
    }
  }
}
