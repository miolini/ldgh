import test from 'node:test';
import assert from 'node:assert/strict';
import {WorkflowJournal} from '../workflow-journal.mjs';
import {runToolConversation} from '../agent-workflow.mjs';

class MemoryStorage {
  values=new Map();
  getItem(key){return this.values.get(key)??null;}
  setItem(key,value){this.values.set(key,String(value));}
}
const completion=message=>({choices:[{message}]});
const call=(id,name='add_analyst_fact')=>({id,type:'function',function:{name,arguments:'{}'}});

test('journal survives a new page instance and restores active tool transaction state',()=>{
  const storage=new MemoryStorage();
  let journal=new WorkflowJournal(storage);
  journal.startRun({runId:'run-1',question:'Investigate',messages:[{role:'user',content:'Investigate'}],pendingCalls:null});
  journal.checkpoint({runId:'run-1',phase:'tool_running',messages:[{role:'assistant',tool_calls:[call('c1')]}],pendingCalls:[call('c1')],nextActionIndex:0,question:'Investigate'});
  journal.appendTranscript({role:'user',text:'Investigate'});
  journal=new WorkflowJournal(storage);
  assert.equal(journal.data.activeRun.phase,'tool_running');
  assert.equal(journal.data.activeRun.nextActionIndex,0);
  assert.equal(journal.data.activeRun.pendingCalls[0].id,'c1');
  assert.deepEqual(journal.data.transcript.map(item=>item.text),['Investigate']);
});

test('resume executes a saved pending action before asking the model again and keeps its idempotency key',async()=>{
  const messages=[{role:'user',content:'Add the finding'}];
  const storage=new MemoryStorage(),journal=new WorkflowJournal(storage);
  journal.startRun({runId:'run-2',question:'Add the finding',messages});
  const pending=[call('fact-1')];
  journal.checkpoint({runId:'run-2',phase:'tool_running',messages,pendingCalls:pending,nextActionIndex:0});
  const restored=new WorkflowJournal(storage).data.activeRun;
  let providerCalls=0,actionIds=[];
  const answer=await runToolConversation(restored.messages,{
    runId:restored.runId,resume:restored,
    executeTool:async(_name,_args,_signal,actionId)=>{actionIds.push(actionId);return {saved:true};},
    requestCompletion:async current=>{providerCalls++;assert.equal(current.at(-1).role,'tool');return completion({content:'Done'});},
  });
  assert.equal(answer,'Done');assert.deepEqual(actionIds,['run-2:fact-1']);assert.equal(providerCalls,1);
});

test('each pending action is checkpointed before and after its execution',async()=>{
  const checkpoints=[],messages=[{role:'user',content:'Read and save'}];let step=0;
  await runToolConversation(messages,{
    onCheckpoint:value=>checkpoints.push({phase:value.phase,index:value.nextActionIndex,tools:value.messages.filter(m=>m.role==='tool').length}),
    requestCompletion:async()=>step++?completion({content:'Finished'}):completion({tool_calls:[call('c1'),call('c2','get_record')]}),
    executeTool:async()=>({ok:true}),
  });
  assert.ok(checkpoints.some(item=>item.phase==='tool_running'&&item.index===0));
  assert.ok(checkpoints.some(item=>item.phase==='tool_running'&&item.index===1));
  assert.ok(checkpoints.some(item=>item.phase==='tool_actions'&&item.tools===1));
  assert.ok(checkpoints.some(item=>item.phase==='tool_actions'&&item.tools===2));
});

test('storage quota errors are surfaced instead of silently dropping a checkpoint',()=>{
  const storage={getItem:()=>null,setItem:()=>{throw new Error('quota');}};
  const journal=new WorkflowJournal(storage);
  assert.throws(()=>journal.startRun({runId:'x'}),/quota/);
});
