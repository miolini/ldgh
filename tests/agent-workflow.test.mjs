import test from 'node:test';
import assert from 'node:assert/strict';
import {runToolConversation,settleConversationTurn,trimConversationHistory,compactConversation,visibleAnswer} from '../agent-workflow.mjs';

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

test('finalizes at the 100-call limit without executing extra actions or appending unmatched history',async()=>{
  const messages=[{role:'user',content:'Search and summarize.'}];let index=0;let executed=0;
  const answer=await runToolConversation(messages,{
    requestCompletion:async(current,signal,round,options)=>index++===0
      ?completion({tool_calls:Array.from({length:100},(_,i)=>call(`first-${i}`))})
      :(assert.equal(options.allowTools,false),assert.equal(current[0].role,'system'),assert.equal(current.filter(item=>item.role==='system').length,1),completion({content:'Completed the first 100 actions.'})),
    executeTool:async()=>{executed++;return {ok:true};},
  });
  assert.equal(answer,'Completed the first 100 actions.');
  assert.equal(executed,100);
  assert.equal(messages.length,102);
  assert.equal(messages.filter(item=>item.role==='tool').length,100);
  assert.deepEqual(messages.at(-1).role,'tool');
});

test('repeated identical case searches are blocked and the agent is forced to summarize',async()=>{
  const messages=[{role:'system',content:'Case snapshot'},{role:'user',content:'Is there evidence of fraud?'}];
  const search=call('search-1','search_case_records',JSON.stringify({query:'fraud indicators',type:'any',limit:10}));
  let request=0,executed=0;
  const answer=await runToolConversation(messages,{
    requestCompletion:async(current,signal,round,options)=>{
      request++;
      if(request===1)return completion({tool_calls:[search]});
      if(request===2)return completion({tool_calls:[{...search,id:'search-2'}]});
      assert.equal(options.allowTools,false);
      assert.equal(current[0].role,'system');
      assert.equal(current.filter(item=>item.role==='system').length,1);
      return completion({content:'The available records do not establish fraud.'});
    },
    executeTool:async()=>{executed++;return [{id:'NOTE-1',title:'Review note'}];},
  });
  assert.equal(answer,'The available records do not establish fraud.');
  assert.equal(executed,1);
  const results=messages.filter(item=>item.role==='tool');
  assert.equal(results.length,2);
  assert.equal(JSON.parse(results[1].content).repeated,true);
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


test('reasoning-only cutoff retries with more space and returns a visible answer',async()=>{
  const messages=[{role:'user',content:'Investigate the case'}],budgets=[];let index=0;
  const answer=await runToolConversation(messages,{
    requestCompletion:async(current,signal,round,options)=>{
      budgets.push(options.maxTokens);
      return index++===0?{choices:[{finish_reason:'length',message:{content:'<think>unfinished'}}]}:completion({content:'<think>hidden</think>Useful findings.'});
    },executeTool:async()=>{throw Error('Unexpected action');},
  });
  assert.equal(answer,'Useful findings.');assert.ok(budgets[1]>budgets[0]);
  assert.deepEqual(messages,[{role:'user',content:'Investigate the case'}]);
});

test('an empty answer after actions retries without repeating the mutation',async()=>{
  const messages=[{role:'user',content:'Add a fact and explain.'}];let index=0,executed=0;
  const answers=[completion({tool_calls:[call('save','add_fact')]}),completion({content:null}),completion({content:'Saved the fact with ID 42.'})];
  const answer=await runToolConversation(messages,{requestCompletion:async(current,signal,round,options)=>{
    if(index===2)assert.equal(options.allowTools,false);
    return answers[index++];
  },executeTool:async()=>{executed++;return {fact:{id:'42',text:'User-supplied observation'}};}});
  assert.equal(executed,1);assert.equal(answer,'Saved the fact with ID 42.');
  assert.equal(messages.filter(m=>m.role==='tool').length,1);
});

test('length-limited answers continue and retain the complete final text',async()=>{
  const messages=[{role:'user',content:'Give a long report.'}];let index=0;const partials=[];
  const answer=await runToolConversation(messages,{requestCompletion:async(current,signal,round,options)=>{
    options.onContent(index?'Second half.':'First half.');
    if(index)assert.equal(options.allowTools,false);
    return {choices:[{finish_reason:index++?'stop':'length',message:{content:index===1?'First half.':'Second half.'}}]};
  },onAnswer:text=>partials.push(text),executeTool:async()=>({})});
  assert.equal(answer,'First half.\nSecond half.');assert.equal(partials.at(-1),answer);
  assert.deepEqual(messages,[{role:'user',content:'Give a long report.'}]);
});

test('truncated tool arguments are regenerated before any action is executed',async()=>{
  const messages=[{role:'user',content:'Save this'}];let index=0,executed=0;
  const answer=await runToolConversation(messages,{requestCompletion:async()=>{
    if(index++===0)return {choices:[{finish_reason:'length',message:{tool_calls:[call('broken','add_fact','{"text":')]}}]};
    return index===2?completion({tool_calls:[call('complete','add_fact','{"text":"Observed"}')]}):completion({content:'Saved.'});
  },executeTool:async(name,args)=>{executed++;assert.equal(args.text,'Observed');return {ok:true};}});
  assert.equal(answer,'Saved.');assert.equal(executed,1);assert.equal(messages.at(-1).tool_call_id,'complete');
});

test('persistent empty completions return actual work instead of discarding results',async()=>{
  const messages=[{role:'user',content:'Update and explain.'}];let index=0,executed=0;
  const answer=await runToolConversation(messages,{requestCompletion:async()=>index++===0?completion({tool_calls:[call('mutation','update_case_assumption')]}):completion({content:'<think>unfinished'}),
    executeTool:async()=>{executed++;return {updated:true,key:'wages',value:100000};}});
  assert.equal(executed,1);assert.match(answer,/wages = 100000/);assert.match(answer,/did not finish/);assert.ok(index<=5);
});

test('a provider ignoring tools-disabled finalization cannot cause an infinite loop',async()=>{
  const messages=[{role:'user',content:'Work'}];let requests=0,executed=0;
  const answer=await runToolConversation(messages,{maxToolCalls:1,requestCompletion:async()=>{requests++;return {choices:[{finish_reason:'length',message:{tool_calls:[call('partial')]}}]};},executeTool:async()=>{executed++;return {};}});
  assert.equal(executed,0);assert.ok(requests<=8);assert.match(answer,/retry/i);
});

test('context compaction preserves the question, source context, completed action and valid pairs',()=>{
  const messages=[{role:'system',content:'Immutable case prefix'},{role:'user',content:'Original investigation request'}];
  for(let i=0;i<30;i++)messages.push({role:'assistant',content:null,tool_calls:[call(`c${i}`,i===0?'update_case_assumption':'search_case_records')]},
    {role:'tool',tool_call_id:`c${i}`,name:i===0?'update_case_assumption':'search_case_records',content:JSON.stringify(i===0?{key:'wages',value:150000}:Array.from({length:10},(_,j)=>({id:`r${i}-${j}`,title:'Source record',description:'x'.repeat(1000)})))});
  const original=JSON.stringify(messages),view=compactConversation(messages,{maxChars:18000});
  assert.equal(JSON.stringify(messages),original);assert.equal(view[0].content,'Immutable case prefix');
  assert.ok(view.some(m=>m.role==='user'&&m.content==='Original investigation request'));
  assert.ok(JSON.stringify(view).includes('150000'));
  assert.ok(JSON.stringify(view).length<18000);
  for(let i=0;i<view.length;i++)if(view[i].role==='tool')assert.ok(view.slice(0,i).some(m=>m.tool_calls?.some(c=>c.id===view[i].tool_call_id)));
});

test('many large parallel tool results stay bounded and paired on the next request',()=>{
  const messages=[{role:'user',content:'Read sources'},{role:'assistant',content:null,tool_calls:Array.from({length:100},(_,i)=>call(`b${i}`))},
    ...Array.from({length:100},(_,i)=>({role:'tool',name:'read_case',tool_call_id:`b${i}`,content:JSON.stringify({details:'x'.repeat(6000)})}))];
  const view=compactConversation(messages,{maxChars:50000});
  assert.ok(JSON.stringify(view).length<50000);assert.equal(view.filter(m=>m.role==='tool').length,100);
});

test('all reasoning blocks are hidden, including a final unclosed block',()=>{
  assert.equal(visibleAnswer('<think>private</think>Answer <THINK>more</THINK>done<think>unfinished'),'Answer done');
});

test('quote-heavy tool data is bounded after JSON escaping',async()=>{
  const messages=[{role:'user',content:'Read'}];let index=0;
  await runToolConversation(messages,{requestCompletion:async()=>index++?completion({content:'Done'}):completion({tool_calls:[call('quote')]}),executeTool:async()=>({data:'"\\'.repeat(20000)})});
  const content=messages.find(m=>m.role==='tool').content;assert.ok(content.length<=6000);assert.ok(JSON.parse(content).truncated);
});
