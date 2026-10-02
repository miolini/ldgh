const MAX_TOOL_RESULT_CHARS = 6000;
const MAX_TOOL_CALLS_PER_REPLY = 100;
const MAX_TOOL_ROUNDS = MAX_TOOL_CALLS_PER_REPLY + 1;
const MAX_HISTORY_MESSAGES = 280;

function serializeToolResult(value) {
  let serialized;
  try { serialized = JSON.stringify(value); }
  catch { serialized = JSON.stringify({error:'Tool result could not be serialized.'}); }
  if (typeof serialized !== 'string') serialized = 'null';
  if (serialized.length <= MAX_TOOL_RESULT_CHARS) return serialized;
  const previewLength = MAX_TOOL_RESULT_CHARS - 100;
  return JSON.stringify({truncated:true,preview:serialized.slice(0,previewLength),note:'Tool result shortened to fit the conversation context.'});
}

function contentText(content) {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) return content.filter(part=>part?.type==='text').map(part=>part.text??'').join('\n');
  return '';
}

export async function runToolConversation(messages, {
  requestCompletion,
  executeTool,
  onStatus=()=>{},
  onToolResult=()=>{},
  signal,
  maxRounds=MAX_TOOL_ROUNDS,
  maxToolCalls=MAX_TOOL_CALLS_PER_REPLY,
}={}) {
  let callsMade=0;
  const usedIds=new Set(messages.flatMap(message=>[
    ...(message.tool_calls??[]).map(call=>call.id),
    ...(message.tool_call_id?[message.tool_call_id]:[]),
  ]).filter(Boolean).map(String));
  for(let round=0;round<maxRounds;round++){
    if(signal?.aborted)throw signal.reason??new DOMException('The operation was aborted.','AbortError');
    try{onStatus(round?`Preparing response from case tool results… (${round+1}/${maxRounds})`:'Asking the local agent…');}catch{}
    const payload=await requestCompletion(messages,signal,round);
    const assistant=payload?.choices?.[0]?.message;
    if(!assistant||typeof assistant!=='object')throw new Error('Local agent returned an invalid response.');
    const calls=Array.isArray(assistant.tool_calls)?assistant.tool_calls:[];
    if(!calls.length)return contentText(assistant.content);
    if(calls.length>maxToolCalls)throw new Error(`Local agent exceeded the maximum of ${maxToolCalls} tool actions for one reply.`);

    const normalizedCalls=calls.map((call,index)=>{
      if(!call||typeof call!=='object'||!call.function||typeof call.function!=='object')throw new Error('Local agent returned an invalid tool action.');
      let id=typeof call.id==='string'?call.id.trim():'';
      if(!id||usedIds.has(id)){
        const base=`local-tool-${round+1}-${index+1}`;
        id=base;let suffix=1;
        while(usedIds.has(id))id=`${base}-${suffix++}`;
      }
      usedIds.add(id);
      return {...call,id};
    });
    if(callsMade+normalizedCalls.length>maxToolCalls)throw new Error(`Local agent exceeded the maximum of ${maxToolCalls} tool actions for one reply.`);
    messages.push({role:'assistant',content:assistant.content??null,tool_calls:normalizedCalls});
    for(const call of normalizedCalls){
      const name=typeof call.function.name==='string'?call.function.name:'';
      let result;
      if(signal?.aborted)result={error:'Action not run because the reply was stopped.'};
      else{
        callsMade++;
        try{
          let args={};
          const rawArgs=call.function.arguments;
          if(typeof rawArgs==='string'){
            try{args=JSON.parse(rawArgs||'{}');}
            catch{throw new Error('The local agent supplied invalid tool arguments.');}
          }
          else if(rawArgs&&typeof rawArgs==='object'&&!Array.isArray(rawArgs))args=rawArgs;
          else if(rawArgs!==undefined&&rawArgs!==null)throw new Error('The local agent supplied invalid tool arguments.');
          result=await executeTool(name,args,signal);
          if(result===undefined)throw new Error('Case tools are still loading. Try again shortly.');
        }catch(cause){result={error:cause?.message??String(cause)};}
      }
      try{onToolResult(name,result);}catch{}
      const toolMessage={role:'tool',tool_call_id:call.id,content:serializeToolResult(result)};
      if(name)toolMessage.name=name;
      messages.push(toolMessage);
    }
  }
  throw new Error('Local agent reached the maximum tool-call rounds without producing a final answer.');
}

function messageSize(message) {
  try { return JSON.stringify(message).length; }
  catch { return 1000; }
}

export function trimConversationHistory(messages,{maxTurns=8,maxChars=48000,maxMessages=MAX_HISTORY_MESSAGES}={}){
  const turns=[];let turn=[];
  for(const message of messages){
    if(message.role==='user'&&turn.length){turns.push(turn);turn=[];}
    turn.push(message);
  }
  if(turn.length)turns.push(turn);
  const kept=[];let size=0,count=0;
  for(let i=turns.length-1;i>=0&&kept.length<maxTurns;i--){
    const turnSize=turns[i].reduce((sum,message)=>sum+messageSize(message),0);
    if(kept.length&&(size+turnSize>maxChars||count+turns[i].length>maxMessages))break;
    kept.unshift(turns[i]);size+=turnSize;count+=turns[i].length;
  }
  return kept.flat();
}

export function settleConversationTurn(history,turnMessages,assistantContent){
  return trimConversationHistory([...history,...turnMessages,{role:'assistant',content:assistantContent}]);
}
