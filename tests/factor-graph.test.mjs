import test from 'node:test';
import assert from 'node:assert/strict';
import { defaultAssumptions, assumptionControls, validateAssumptionUpdate, normalizeAssumptions } from '../model-config.mjs';
import { simulateCase } from '../probabilistic-model.mjs';
import { buildFactorGraph } from '../factor-graph.mjs';

test('expanded graph has valid links, readable bounds and editable node mappings',()=>{
  const result=simulateCase(defaultAssumptions,[],600);
  const graph=buildFactorGraph(defaultAssumptions,[],result);
  const ids=new Set(graph.nodes.map(n=>n.id)),keys=new Set(assumptionControls.map(c=>c.key));
  assert.ok(graph.nodes.length>=30);
  assert.equal(ids.size,graph.nodes.length);
  for(const edge of graph.edges){assert.ok(ids.has(edge.from));assert.ok(ids.has(edge.to));}
  for(const node of graph.nodes){
    assert.ok(node.x+node.w<=graph.width&&node.y+node.h<=graph.height);
    if(node.key)assert.ok(keys.has(node.key),`${node.id} points to an editable control`);
    assert.ok(!node.metric.includes('NaN'));
  }
});

test('ranges reject contradictions without changing the existing scenario',()=>{
  assert.throws(()=>validateAssumptionUpdate(defaultAssumptions,'futureLow',100000));
  assert.throws(()=>validateAssumptionUpdate(defaultAssumptions,'settlement',.99));
  assert.throws(()=>validateAssumptionUpdate(defaultAssumptions,'nonEconomicLow',3));
  assert.throws(()=>validateAssumptionUpdate(defaultAssumptions,'medicalCausation',NaN));
  assert.equal(validateAssumptionUpdate(defaultAssumptions,'futureLow',20000).futureLow,20000);
  assert.equal(defaultAssumptions.futureLow,18000);
});


test('older saved inputs retain their resolution mode during expansion',()=>{
  const migrated=normalizeAssumptions({settlement:.95});
  assert.equal(migrated.settlement,.95);assert.equal(migrated.settlementHigh,1);
  assert.ok(Object.values(normalizeAssumptions({settlement:'bad'})).every(Number.isFinite));
});
