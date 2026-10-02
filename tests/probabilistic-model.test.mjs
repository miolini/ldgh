import test from 'node:test';
import assert from 'node:assert/strict';
import { evidenceLikelihoodRatio, posterior, simulateCase } from '../probabilistic-model.mjs';

function seeded(seed=1){let value=seed>>>0;return()=>{value=(1664525*value+1013904223)>>>0;return value/4294967296;};}
const assumptions={liability:.62,scope:.62,liabilityScopeDependence:0,specialsSupport:.72,wageSupport:.38,surgery:.55,
  specials:118400,wages:214000,futureLow:18000,futureHigh:65000,nonEconomicLow:.7,nonEconomicMode:1.45,nonEconomicHigh:2.8,
  comparativeLow:.05,comparativeHigh:.35,settlement:.72,coverageLimit:100000,lien:22180,noFaultPaid:50000};

test('repeated facts from the same source are partially discounted',()=>{
  const one=[{id:'a',source:'Provider Ledger · Sep 1',dimension:'liability',lr:2,confidence:1}];
  const same=[...one,{id:'b',source:'provider ledger sep 1',dimension:'liability',lr:2,confidence:1}];
  const separate=[...one,{id:'b',source:'Independent witness · Sep 2',dimension:'liability',lr:2,confidence:1}];
  assert.equal(evidenceLikelihoodRatio(one,'liability'),2);
  assert.ok(evidenceLikelihoodRatio(same,'liability')>2);
  assert.ok(evidenceLikelihoodRatio(same,'liability')<evidenceLikelihoodRatio(separate,'liability'));
});

test('evidence likelihood ratios update posterior in the expected direction',()=>{
  const prior=.5;
  assert.ok(posterior(prior,[{source:'s',dimension:'liability',lr:2,confidence:1}],'liability')>.5);
  assert.ok(posterior(prior,[{source:'s',dimension:'liability',lr:.5,confidence:1}],'liability')<.5);
});

test('simulation is reproducible and ranks high-impact uncertain factors',()=>{
  const a=simulateCase(assumptions,[],1200,seeded(22));
  const b=simulateCase(assumptions,[],1200,seeded(22));
  assert.deepEqual([...a.outcomes],[...b.outcomes]);
  assert.deepEqual(a.focus.map(({key,priorityScore})=>[key,priorityScore]),b.focus.map(({key,priorityScore})=>[key,priorityScore]));
  assert.equal(a.focus.length,6);
  assert.ok(a.focus.every(item=>item.priorityScore>=0&&item.priorityScore<=100&&item.conditionalSwing>=0));
  assert.ok(a.focus.some(item=>item.key==='scope'));
  assert.ok(a.zeroRate>=0&&a.zeroRate<=1);
  assert.ok(a.policyLimitRate>=0&&a.policyLimitRate<=1);
});

test('higher shared-rank dependence changes the joint liability gate rate',()=>{
  const independent=simulateCase({...assumptions,liabilityScopeDependence:0},[],5000,seeded(9));
  const dependent=simulateCase({...assumptions,liabilityScopeDependence:.8},[],5000,seeded(9));
  assert.ok(dependent.liabilityRate>independent.liabilityRate+.1);
});
