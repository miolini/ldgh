import test from 'node:test';
import assert from 'node:assert/strict';
import { evidenceLikelihoodRatio, posterior, simulateCase, mergeNodeStats, summarizeNodeStats } from '../probabilistic-model.mjs';

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
  assert.equal(a.focus.length,15);
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


test('shared causation changes clinical, income and care branches together',()=>{
  const low=simulateCase({...assumptions,medicalCausation:.01},[],12000,seeded(7));
  const high=simulateCase({...assumptions,medicalCausation:.99},[],12000,seeded(7));
  for(const key of ['pastMedical','supportedWages','futureCare'])assert.ok(high.nodeEstimates[key].mean>low.nodeEstimates[key].mean*10,key);
  assert.ok(high.mean>low.mean*10);
});

test('surgery necessity evidence changes future care, without updating unrelated priors',()=>{
  const facts=[{id:'opinion',source:'Treating opinion',dimension:'surgeryNecessity',lr:4,confidence:1}];
  const before=simulateCase(assumptions,[],8000,seeded(10));
  const after=simulateCase(assumptions,facts,8000,seeded(10));
  assert.ok(after.nodeEstimates.futureCare.mean>before.nodeEstimates.futureCare.mean);
  assert.equal(after.probabilities.scope,before.probabilities.scope);
  assert.equal(after.probabilities.surgery,before.probabilities.surgery);
});

test('collection and time discount preserve payout bounds',()=>{
  const immediate=simulateCase({...assumptions,collectability:.99,annualDiscount:0},[],8000,seeded(17));
  const delayed=simulateCase({...assumptions,collectability:.99,annualDiscount:.2,resolutionMonthsLow:24,resolutionMonthsHigh:24},[],8000,seeded(17));
  assert.ok(delayed.mean<immediate.mean);
  assert.ok(immediate.outcomes.every(v=>v>=0&&v<=assumptions.coverageLimit-assumptions.lien));
  assert.ok(Math.abs(immediate.nodeEstimates.net.mean-immediate.mean)<1e-8);
  assert.ok(Math.abs(delayed.nodeEstimates.net.mean/1.2**2-delayed.mean)<1e-8);
  const blocked=simulateCase({...assumptions,collectability:.01},[],8000,seeded(17));
  assert.ok(blocked.mean<immediate.mean*.05);
});

test('greater concentration reduces assumed parameter uncertainty without asserting accuracy',()=>{
  const broad=simulateCase({...assumptions,probabilityConcentration:2,medicalConcentration:2},[],8000,seeded(8));
  const narrow=simulateCase({...assumptions,probabilityConcentration:200,medicalConcentration:200},[],8000,seeded(8));
  assert.ok(narrow.nodeEstimates.medicalCausation.std<broad.nodeEstimates.medicalCausation.std/3);
  assert.ok(narrow.nodeEstimates.specialsSupport.std<broad.nodeEstimates.specialsSupport.std/3);
  assert.equal(narrow.probabilities.medicalCausation,broad.probabilities.medicalCausation);
});

test('node statistics aggregate all batches and reconcile supported damage totals',()=>{
  const a=simulateCase(assumptions,[],1000,seeded(91)),b=simulateCase(assumptions,[],2500,seeded(3));
  const totals={};mergeNodeStats(totals,a.nodeStats);mergeNodeStats(totals,b.nodeStats);
  const combined=summarizeNodeStats(totals);
  for(const key of Object.keys(combined))assert.ok(Math.abs(combined[key].mean-(a.nodeEstimates[key].mean*1000+b.nodeEstimates[key].mean*2500)/3500)<1e-7,key);
  assert.equal(totals.medicalCausation.histogram.reduce((a,b)=>a+b,0),3500);
  assert.ok(Math.abs(combined.damages.mean-['pastMedical','supportedWages','futureCare','noneconomic'].reduce((sum,key)=>sum+combined[key].mean,0))<1e-7);
  assert.ok(Object.values(combined).every(s=>Number.isFinite(s.mean)&&s.p10<=s.p90));
});


test('zero-mass node quantiles remain zero and invalid fact weights are neutral',()=>{
  const result=simulateCase({...assumptions,medicalCausation:.01},[],1000,seeded(7));
  assert.equal(result.nodeEstimates.net.p10,0);
  assert.equal(result.nodeEstimates.futureCare.p10,0);
  assert.equal(evidenceLikelihoodRatio([{source:'bad',dimension:'liability',lr:NaN,confidence:1}],'liability'),1);
});
