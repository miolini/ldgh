import test from 'node:test';
import assert from 'node:assert/strict';

test('one million trial worker yields progress and returns full-run statistics',async()=>{
  const events=[];
  globalThis.self={postMessage:message=>events.push(message)};
  await import('../simulation-worker.mjs');
  const assumptions={liability:.62,scope:.62,liabilityScopeDependence:0,specialsSupport:.72,wageSupport:.38,surgery:.55,
    specials:118400,wages:214000,futureLow:18000,futureHigh:65000,nonEconomicLow:.7,nonEconomicMode:1.45,nonEconomicHigh:2.8,
    comparativeLow:.05,comparativeHigh:.35,settlement:.72,coverageLimit:100000,lien:22180,noFaultPaid:50000};
  const run=self.onmessage({data:{assumptions,facts:[]}});
  await new Promise(resolve=>setTimeout(resolve,0));
  assert.ok(events.some(event=>event.type==='progress'&&event.completed>0),'progress should be observable before completion');
  assert.ok(!events.some(event=>event.complete),'the full run should still be in progress');
  await run;
  const results=events.filter(event=>event.type==='result');
  const final=results.at(-1)?.result;
  assert.equal(results.length,10);
  assert.equal(final.iterations,1_000_000);
  assert.equal(final.outcomes.length,1_000_000);
  assert.ok(final.outcomes.every(Number.isFinite));
  assert.ok(Math.abs(final.outcomes.reduce((sum,value)=>sum+value,0)/final.iterations-final.mean)<1e-6);
  for(const factor of final.focus){
    const weightedMean=factor.meanWithFactor*factor.probability+factor.meanWithoutFactor*(1-factor.probability);
    assert.ok(Math.abs(weightedMean-final.mean)<1e-6,`${factor.label} conditional means should reconcile to the overall mean`);
  }
  for(const event of results){
    const nodes=event.result.nodeEstimates;
    assert.ok(Math.abs(nodes.presentValue.mean-event.result.mean)<1e-6,'node results must use the full cumulative run');
    assert.ok(Math.abs(nodes.damages.mean-['pastMedical','supportedWages','futureCare','noneconomic'].reduce((sum,key)=>sum+nodes[key].mean,0))<1e-6);
  }
  delete globalThis.self;
});
