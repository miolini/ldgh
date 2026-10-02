import { simulateCase } from './probabilistic-model.mjs';

export const SIMULATION_ITERATIONS=1_000_000;
const BATCH_SIZE=10_000;
const SUMMARY_INTERVAL=100_000;
const PREVIEW_SIZE=10_000;
const percentile=(sorted,fraction)=>sorted[Math.min(sorted.length-1,Math.floor((sorted.length-1)*fraction))]??0;

self.onmessage=({data:{assumptions,facts}})=>{
  try {
    const outcomes=new Float64Array(SIMULATION_ITERATIONS);
    const focusTotals=new Map();
    let completed=0,sum=0,liableCount=0,zeroCount=0,policyLimitCount=0;
    while(completed<SIMULATION_ITERATIONS){
      const batchSize=Math.min(BATCH_SIZE,SIMULATION_ITERATIONS-completed);
      const batch=simulateCase(assumptions,facts,batchSize);
      outcomes.set(batch.outcomes,completed);
      completed+=batchSize;
      sum+=batch.mean*batchSize;
      liableCount+=Math.round(batch.liabilityRate*batchSize);
      zeroCount+=Math.round(batch.zeroRate*batchSize);
      policyLimitCount+=Math.round(batch.policyLimitRate*batchSize);
      for(const item of batch.focus){
        const cell=focusTotals.get(item.key)??{definition:item,yesCount:0,noCount:0,yesSum:0,noSum:0};
        const yesCount=Math.round(item.probability*batchSize),noCount=batchSize-yesCount;
        cell.yesCount+=yesCount;cell.noCount+=noCount;
        cell.yesSum+=item.meanWithFactor*yesCount;cell.noSum+=item.meanWithoutFactor*noCount;
        focusTotals.set(item.key,cell);
      }
      self.postMessage({type:'progress',completed,total:SIMULATION_ITERATIONS});
      if(completed%SUMMARY_INTERVAL!==0&&completed!==SIMULATION_ITERATIONS)continue;
      const complete=completed===SIMULATION_ITERATIONS;
      const sorted=complete?outcomes:outcomes.slice(0,completed);
      sorted.sort();
      const positiveCount=completed-zeroCount;
      const conditionalP50=positiveCount?sorted[zeroCount+Math.floor((positiveCount-1)*.5)]:0;
      const focus=[...focusTotals.values()].map(cell=>{
        const probability=cell.yesCount/completed;
        const meanWithFactor=cell.yesCount?cell.yesSum/cell.yesCount:0;
        const meanWithoutFactor=cell.noCount?cell.noSum/cell.noCount:0;
        const conditionalSwing=Math.abs(meanWithFactor-meanWithoutFactor);
        return {...cell.definition,probability,meanWithFactor,meanWithoutFactor,conditionalSwing,
          focusValue:conditionalSwing*4*probability*(1-probability)};
      }).sort((a,b)=>b.focusValue-a.focusValue);
      const maxFocus=focus[0]?.focusValue||1;
      for(const item of focus)item.priorityScore=Math.round(item.focusValue/maxFocus*100);
      // Exact statistics use all trials; a small quantile sample keeps live chart updates cheap.
      const chartOutcomes=complete?sorted:Float64Array.from({length:Math.min(PREVIEW_SIZE,completed)},(_,index)=>sorted[Math.floor(index*(completed-1)/(Math.min(PREVIEW_SIZE,completed)-1))]);
      const result={...batch,outcomes:chartOutcomes,iterations:completed,mean:sum/completed,
        liabilityRate:liableCount/completed,zeroRate:zeroCount/completed,policyLimitRate:policyLimitCount/completed,
        p10:percentile(sorted,.1),p50:percentile(sorted,.5),p90:percentile(sorted,.9),conditionalP50,focus};
      self.postMessage({type:'result',result,complete},[chartOutcomes.buffer]);
    }
  } catch(error){self.postMessage({type:'error',message:error?.message??String(error)});}
};
