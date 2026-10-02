import test from 'node:test';
import assert from 'node:assert/strict';
import {runToolConversation,settleConversationTurn,trimConversationHistory} from '../agent-workflow.mjs';

const completion=message=>({choices:[{message}]});
const call=(id,name='read_case',args='{}')=>({id,type:'function',function:{name,arguments:args}});

test('tool cycles stay paired in the next request and complete a multi-turn exchange',async()=>{
  const messages=[{role:'user',content:'Find the next deadline.'}];
  const sent=[];let responseIndex=0;
  const answer=await runToolConversation(messages,{
    requestCompletion:async current=>{
      sent.push(structuredClone(current));
      return responseIndex++===0?completion({content:null,tool_calls:[call('call-1')] }):completion({content:'The next deadline is October 5.'});
    },
    executeTool:async()=>({deadline:'October 5'}),
  });
  assert.equal(answer,'The next deadline is October 5.');
  assert.deepEqual(messages.map(item=>item.role),['user','assistant','tool']);
  assert.deepEqual(sent[1].slice(-2).map(item=>item.role),['assistant','tool']);
  assert.equal(sent[1].at(-1).tool_call_id,sent[1].at(-2).tool_calls[0].id);
  const history=settleConversationTurn([],messages,answer);
  const nextRequest=[{role:'system',content:'Case context.'},...history,{role:'user',content:'And who authored it?'}];
  assert.deepEqual(nextRequest.slice(-5).map(item=>item.role),['user','assistant','tool','assistant','user']);
  assert.equal(nextRequest.at(-2).content,answer);
});

test('allows 100 tool actions in one batch and pairs all results',async()=>{
  const messages=[{role:'user',content:'Read records.'}];let index=0;
  const calls=Array.from({length:100},(_,i)=>call(i===0?undefined:`id-${i}`));
  let executed=0;
  const answer=await runToolConversation(messages,{
    requestCompletion:async()=>index++===0?completion({tool_calls:calls}):completion({content:'Done.'}),
    executeTool:async()=>{executed++;return {ok:true};},
  });
  assert.equal(answer,'Done.');
  assert.equal(executed,100);
  const request=messages.find(item=>item.role==='assistant');
  const results=messages.filter(item=>item.role==='tool');
  assert.equal(request.tool_calls.length,100);
  assert.equal(new Set(request.tool_calls.map(item=>item.id)).size,100);
  assert.deepEqual(results.map(item=>item.tool_call_id),request.tool_calls.map(item=>item.id));
  assert.deepEqual(JSON.parse(results.at(-1).content),{ok:true});
});

test('supports 100 serialized tool rounds followed by a final answer',async()=>{
  const messages=[{role:'user',content:'Review every record.'}];let round=0;let executed=0;
  const answer=await runToolConversation(messages,{
    requestCompletion:async()=>round<100?completion({tool_calls:[call(`serial-${round++}`)]}):completion({content:'All 100 records were reviewed.'}),
    executeTool:async()=>{executed++;return {read:true};},
  });
  assert.equal(answer,'All 100 records were reviewed.');
  assert.equal(round,100);
  assert.equal(executed,100);
  assert.equal(messages.filter(item=>item.role==='tool').length,100);
});

test('rejects actions beyond the 100-call limit without appending unmatched history',async()=>{
  const messages=[{role:'user',content:'Search and summarize.'}];let index=0;let executed=0;
  await assert.rejects(runToolConversation(messages,{
    requestCompletion:async()=>index++===0
      ?completion({tool_calls:Array.from({length:100},(_,i)=>call(`first-${i}`))})
      :completion({tool_calls:[call('over-limit')]}),
    executeTool:async()=>{executed++;return {ok:true};},
  }),/exceeded the maximum/);
  assert.equal(executed,100);
  assert.equal(messages.length,102);
  assert.equal(messages.filter(item=>item.role==='tool').length,100);
  assert.deepEqual(messages.at(-1).role,'tool');
});

test('invalid arguments become a matched tool error instead of breaking the protocol',async()=>{
  const messages=[{role:'user',content:'Read this.'}];let index=0;let executed=0;
  await runToolConversation(messages,{
    requestCompletion:async()=>index++===0?completion({tool_calls:[call('bad-args','read_case','{')] }):completion({content:'Recovered.'}),
    executeTool:async()=>{executed++;return {};},
  });
  assert.equal(executed,0);
  assert.equal(messages.at(-1).tool_call_id,'bad-args');
  assert.match(JSON.parse(messages.at(-1).content).error,/invalid tool arguments/i);
});

test('tool result payloads are bounded and remain valid JSON',async()=>{
  const messages=[{role:'user',content:'Read a large result.'}];let index=0;
  await runToolConversation(messages,{
    requestCompletion:async()=>index++===0?completion({tool_calls:[call('large')] }):completion({content:'Done.'}),
    executeTool:async()=>({record:'x'.repeat(20000)}),
  });
  const result=messages.find(item=>item.role==='tool').content;
  assert.ok(result.length<=6000);
  assert.equal(JSON.parse(result).truncated,true);
});

test('stopping during a requested action preserves its result and aborts the next round',async()=>{
  const controller=new AbortController();
  const messages=[{role:'user',content:'Read this.'}];let index=0;let executed=0;
  await assert.rejects(runToolConversation(messages,{
    requestCompletion:async()=>index++===0?completion({tool_calls:[call('stop-me')] }):completion({content:'Stopped safely.'}),
    executeTool:async()=>{executed++;controller.abort();return {ok:true};},
    signal:controller.signal,
  }),{name:'AbortError'});
  assert.equal(executed,1);
  assert.equal(messages.at(-1).tool_call_id,'stop-me');
});

test('history pruning keeps complete recent turns and obeys turn and size limits',()=>{
  const history=[];
  for(let i=0;i<4;i++)history.push(
    {role:'user',content:`question-${i}`},
    {role:'assistant',content:null,tool_calls:[call(`call-${i}`)]},
    {role:'tool',tool_call_id:`call-${i}`,content:`result-${i}`},
    {role:'assistant',content:`answer-${i}`},
  );
  const pruned=trimConversationHistory(history,{maxTurns:2,maxChars:10000});
  assert.deepEqual(pruned.filter(item=>item.role==='user').map(item=>item.content),['question-2','question-3']);
  for(let i=0;i<pruned.length;i++)if(pruned[i].role==='tool')assert.equal(pruned[i-1].tool_calls[0].id,pruned[i].tool_call_id);
  const bySize=trimConversationHistory(history,{maxTurns:8,maxChars:100});
  assert.equal(bySize[0].content,'question-3');
  assert.equal(bySize.at(-1).content,'answer-3');
  const crowded=Array.from({length:5},(_,turnIndex)=>[
    {role:'user',content:`q${turnIndex}`},
    ...Array.from({length:80},(_,index)=>({role:'assistant',content:`m${turnIndex}-${index}`})),
  ]).flat();
  const byCount=trimConversationHistory(crowded,{maxTurns:8,maxChars:100000,maxMessages:100});
  assert.equal(byCount.filter(item=>item.role==='user').length,1);
  assert.equal(byCount[0].content,'q4');
});
