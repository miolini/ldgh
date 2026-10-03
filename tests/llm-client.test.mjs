import test from 'node:test';
import assert from 'node:assert/strict';
import {readCompletionResponse,requestLocalCompletion} from '../llm-client.mjs';
const encoder=new TextEncoder();
const packet=payload=>`data: ${typeof payload==='string'?payload:JSON.stringify(payload)}\r\n\r\n`;
const choice=(delta,finish_reason=null)=>({choices:[{index:0,delta,finish_reason}]});
function stream(text,split=7){const bytes=encoder.encode(text);return new Response(new ReadableStream({start(controller){for(let i=0;i<bytes.length;i+=split)controller.enqueue(bytes.slice(i,i+split));controller.close();}}),{headers:{'content-type':'text/event-stream'}});}

test('streamed UTF-8 content survives arbitrary byte and CRLF boundaries',async()=>{
  const updates=[],response=await readCompletionResponse(stream(packet(choice({content:'Привет '}))+packet(choice({content:'case.'},'stop'))+packet('[DONE]'),1),{onContent:text=>updates.push(text)});
  assert.equal(response.choices[0].message.content,'Привет case.');assert.equal(response.choices[0].finish_reason,'stop');assert.equal(updates.at(-1),'Привет case.');
});

test('fragmented tool calls preserve names, IDs and arguments while hiding reasoning',async()=>{
  const updates=[],frames=[choice({reasoning_content:'private reasoning'}),choice({tool_calls:[{index:0,id:'call_',function:{name:'get_case_',arguments:'{"section":'}}]}),choice({tool_calls:[{index:0,id:'1',function:{name:'model',arguments:'"inputs"}'}}]},'tool_calls')];
  const response=await readCompletionResponse(stream(frames.map(packet).join('')+packet('[DONE]')),{onContent:text=>updates.push(text)});
  assert.deepEqual(response.choices[0].message.tool_calls,[{id:'call_1',type:'function',function:{name:'get_case_model',arguments:'{"section":"inputs"}'}}]);
  assert.equal(updates.length,0);assert.equal(response.choices[0].message.content,null);
});

test('a dropped stream is retried before any tool request is delivered to the workflow',async()=>{
  let requests=0;const payload={model:'local',messages:[],stream:true};
  const response=await requestLocalCompletion(payload,{fetchImpl:async()=>++requests===1?stream(packet(choice({tool_calls:[{index:0,id:'broken',function:{name:'add_fact',arguments:'{}'}}]}))):stream(packet(choice({content:'Recovered.'},'stop'))+packet('[DONE]'))});
  assert.equal(requests,2);assert.equal(response.choices[0].message.content,'Recovered.');assert.equal(response.choices[0].message.tool_calls,undefined);
});

test('JSON fallback and explicit provider errors are handled',async()=>{
  const json={choices:[{finish_reason:'stop',message:{content:'Done'}}]};
  assert.deepEqual(await readCompletionResponse(new Response(JSON.stringify(json),{headers:{'content-type':'application/json'}})),json);
  await assert.rejects(readCompletionResponse(new Response('{"error":{"message":"invalid request"}}',{status:400})),/invalid request/);
  await assert.rejects(readCompletionResponse(stream(packet({error:{message:'provider failed'}}))),/provider failed/);
});

test('stopping aborts retry backoff promptly',async()=>{
  const controller=new AbortController();let requests=0;
  await assert.rejects(requestLocalCompletion({messages:[]},{signal:controller.signal,fetchImpl:async()=>{requests++;return new Response('Unavailable',{status:503});},onRetry:()=>controller.abort()}),{name:'AbortError'});
  assert.equal(requests,1);
});
