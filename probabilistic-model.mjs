import { probabilityFactors, normalizeAssumptions } from './model-config.mjs?build=20261002-expanded-graph';

const clamp=(value,min,max)=>Math.min(max,Math.max(min,Number(value)));

function normal(rng){
  const u=Math.max(Number.EPSILON,rng());
  return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*rng());
}

function gamma(shape,rng){
  if(shape<1)return gamma(shape+1,rng)*Math.pow(rng(),1/shape);
  const d=shape-1/3,c=1/Math.sqrt(9*d);
  for(;;){
    const x=normal(rng),base=1+c*x;
    if(base<=0)continue;
    const v=base*base*base,u=rng();
    if(u<1-.0331*x**4||Math.log(u)<.5*x*x+d*(1-v+Math.log(v)))return d*v;
  }
}

function beta(mean,concentration,rng){
  const p=clamp(mean,.005,.995);
  const a=gamma(p*concentration,rng),b=gamma((1-p)*concentration,rng);
  return a/(a+b);
}

function triangular(low,mode,high,rng){
  if(high<=low)return low;
  const split=(mode-low)/(high-low),u=rng();
  return u<split?low+Math.sqrt(u*(high-low)*(mode-low)):high-Math.sqrt((1-u)*(high-low)*(high-mode));
}

function percentile(sorted,fraction){return sorted[Math.min(sorted.length-1,Math.floor((sorted.length-1)*fraction))]??0;}

export function evidenceLogFactor(facts,dimension){
  const groups=new Map();
  for(const fact of facts){
    if(fact.dimension!==dimension)continue;
    const source=String(fact.source??'').trim().toLocaleLowerCase().replace(/[^\p{L}\p{N}]+/gu,' ');
    const key=source||`unreferenced:${fact.id??groups.size}`;
    if(!Number.isFinite(Number(fact.lr))||!Number.isFinite(Number(fact.confidence)))continue;
    if(!groups.has(key))groups.set(key,[]);
    const lr=clamp(fact.lr,.25,4),confidence=clamp(fact.confidence,0,1);
    groups.get(key).push(Math.log(lr)*confidence);
  }
  let total=0;
  for(const contributions of groups.values())total+=contributions.reduce((sum,value)=>sum+value,0)/Math.sqrt(contributions.length);
  return clamp(total,Math.log(.01),Math.log(100));
}

export function evidenceLikelihoodRatio(facts,dimension){return Math.exp(evidenceLogFactor(facts,dimension));}

export function posterior(prior,facts,dimension){
  const p=clamp(prior,.01,.99),logOdds=Math.log(p/(1-p))+evidenceLogFactor(facts,dimension);
  return 1/(1+Math.exp(-logOdds));
}

const extraFocus=[
  {key:'comparativeFault',label:'Comparative fault',action:'Resolve vehicle positions, sight lines and competing incident accounts.'},
  {key:'wageFraction',label:'Income loss amount',action:'Reconcile the claimed amount against documented earnings and the restriction period.'},
  {key:'resolutionMonths',label:'Resolution timing',action:'Review outstanding discovery and concrete procedural milestones.'},
  {key:'lienPaid',label:'Lien payment',action:'Reconcile the current lien and any documented reduction agreement.'},
];
const focusDefinitions=[...probabilityFactors.map(f=>({...f,dimension:f.key})),...extraFocus.map(f=>({...f,dimension:null}))];
const HISTOGRAM_BINS=128;
export function summarizeNodeStats(stats){
  return Object.fromEntries(Object.entries(stats).map(([key,s])=>{
    const quantile=p=>{
      if(s.min===s.max)return s.min;
      const rank=Math.floor((s.count-1)*p);if(rank<s.zeroCount)return 0;let cumulative=0;
      for(let i=0;i<s.histogram.length;i++){cumulative+=s.histogram[i];if(cumulative>rank)return clamp((i+.5)/s.histogram.length*s.domainMax,s.min,s.max);}
      return s.max;
    };
    const mean=s.sum/s.count;
    return [key,{mean,std:Math.sqrt(Math.max(0,s.sumSquares/s.count-mean*mean)),p10:quantile(.1),p90:quantile(.9),min:s.min,max:s.max,
      eventRate:s.eventCount==null?null:s.eventCount/s.count}];
  }));
}
export function mergeNodeStats(totals,batch){
  for(const [key,s] of Object.entries(batch)){
    const cell=totals[key]??={count:0,zeroCount:0,sum:0,sumSquares:0,min:Infinity,max:-Infinity,domainMax:s.domainMax,histogram:new Float64Array(HISTOGRAM_BINS),eventCount:s.eventCount==null?null:0};
    cell.count+=s.count;cell.zeroCount+=s.zeroCount;cell.sum+=s.sum;cell.sumSquares+=s.sumSquares;cell.min=Math.min(cell.min,s.min);cell.max=Math.max(cell.max,s.max);
    if(cell.eventCount!=null)cell.eventCount+=s.eventCount;
    for(let i=0;i<HISTOGRAM_BINS;i++)cell.histogram[i]+=s.histogram[i];
  }
}

export function simulateCase(assumptions,facts,iterations=1_000_000,rng=Math.random){
  if(!Number.isInteger(iterations)||iterations<1)throw new Error('Iterations must be a positive integer.');
  const a=normalizeAssumptions(assumptions);
  const probabilities=Object.fromEntries(probabilityFactors.map(f=>[f.key,posterior(a[f.key],facts,f.key)]));
  const results=new Float64Array(iterations);
  const accumulators=new Map(focusDefinitions.map(item=>[item.key,{yesCount:0,noCount:0,yesSum:0,noSum:0}]));
  const maxMedical=Math.max(0,a.specials-a.noFaultPaid),maxCare=a.futureHigh,maxWages=a.wages*a.wageFractionHigh;
  const maxNoneconomic=(maxMedical+maxCare)*a.nonEconomicHigh;
  const maxDamages=maxMedical+maxCare+maxWages+maxNoneconomic;
  const domains={...Object.fromEntries(probabilityFactors.map(f=>[f.key,1])),pastMedical:maxMedical,supportedWages:maxWages,futureCare:maxCare,noneconomic:maxNoneconomic,
    damages:maxDamages,comparativeFault:1,wageFraction:1,resolutionFactor:1,resolutionMonths:a.resolutionMonthsHigh,lienPaid:a.lien,
    gross:maxDamages,covered:a.coverageLimit||maxDamages,collected:a.coverageLimit||maxDamages,net:a.coverageLimit||maxDamages,presentValue:a.coverageLimit||maxDamages};
  const nodeStats=Object.fromEntries(Object.entries(domains).map(([key,domainMax])=>[key,{count:0,zeroCount:0,sum:0,sumSquares:0,min:Infinity,max:-Infinity,domainMax:Math.max(1,domainMax),histogram:new Float64Array(HISTOGRAM_BINS),eventCount:probabilities[key]!=null&&key!=='specialsSupport'?0:null}]));
  const nodeEntries=Object.entries(nodeStats);
  let liableCount=0,zeroCount=0,policyLimitCount=0;
  const dependence=a.liabilityScopeDependence;
  for(let i=0;i<iterations;i++){
    const draws=Object.fromEntries(probabilityFactors.map(f=>[f.key,beta(probabilities[f.key],['specialsSupport','treatmentContinuity'].includes(f.key)?a.medicalConcentration:a.probabilityConcentration,rng)]));
    const sharedRank=rng()<dependence,commonRank=rng();
    const events={liability:(sharedRank?commonRank:rng())<draws.liability,scope:(sharedRank?commonRank:rng())<draws.scope};
    for(const f of probabilityFactors)if(f.key!=='specialsSupport'&&!(f.key in events))events[f.key]=rng()<draws[f.key];
    const liable=events.liability&&events.scope;
    // One shared causation event links clinical damages, work loss and future care.
    const pastMedical=events.medicalCausation?Math.max(0,a.specials*draws.specialsSupport*draws.treatmentContinuity-a.noFaultPaid):0;
    const wageFraction=triangular(a.wageFractionLow,a.wageFractionMode,a.wageFractionHigh,rng);
    const supportedWages=events.medicalCausation&&events.wageSupport&&events.wageAttribution?a.wages*wageFraction:0;
    const futureCare=events.medicalCausation&&events.surgeryNecessity&&events.surgery?triangular(a.futureLow,(a.futureLow+a.futureHigh)/2,a.futureHigh,rng):0;
    const lasting=events.medicalCausation&&events.lastingImpairment&&events.treatmentContinuity;
    const noneconomic=(pastMedical+futureCare)*triangular(a.nonEconomicLow,a.nonEconomicMode,a.nonEconomicHigh,rng)*(lasting?1:a.transientMultiplier);
    const damages=pastMedical+supportedWages+futureCare+noneconomic;
    const comparative=triangular(a.comparativeLow,(a.comparativeLow+a.comparativeHigh)/2,a.comparativeHigh,rng);
    const resolutionFactor=triangular(a.settlementLow,a.settlement,a.settlementHigh,rng);
    const gross=liable?damages*(1-comparative)*resolutionFactor:0;
    const policyHit=liable&&a.coverageLimit>0&&gross>=a.coverageLimit;
    const covered=a.coverageLimit>0?Math.min(gross,a.coverageLimit):gross;
    const collected=events.collectability?covered:0;
    const lienPaid=a.lien*triangular(a.lienFractionLow,(a.lienFractionLow+a.lienFractionHigh)/2,a.lienFractionHigh,rng);
    const net=Math.max(0,collected-lienPaid);
    const resolutionMonths=triangular(a.resolutionMonthsLow,(a.resolutionMonthsLow+a.resolutionMonthsHigh)/2,a.resolutionMonthsHigh,rng);
    const presentValue=net/Math.pow(1+a.annualDiscount,resolutionMonths/12);
    results[i]=presentValue;
    liableCount+=liable?1:0;zeroCount+=presentValue===0?1:0;policyLimitCount+=policyHit?1:0;
    const values={...draws,pastMedical,supportedWages,futureCare,noneconomic,damages,comparativeFault:comparative,wageFraction,resolutionFactor,resolutionMonths,lienPaid,gross,covered,collected,net,presentValue};
    for(const [key,cell] of nodeEntries){
      const v=values[key];cell.count++;cell.zeroCount+=v===0?1:0;cell.sum+=v;cell.sumSquares+=v*v;cell.min=Math.min(cell.min,v);cell.max=Math.max(cell.max,v);
      cell.histogram[Math.min(HISTOGRAM_BINS-1,Math.floor(v/cell.domainMax*HISTOGRAM_BINS))]++;
      if(cell.eventCount!=null)cell.eventCount+=events[key]?1:0;
    }
    const statuses={...events,treatmentContinuity:draws.treatmentContinuity>=probabilities.treatmentContinuity,specialsSupport:draws.specialsSupport>=probabilities.specialsSupport,
      comparativeFault:comparative>=(a.comparativeLow+a.comparativeHigh)/2,wageFraction:wageFraction>=a.wageFractionMode,
      resolutionMonths:resolutionMonths>=(a.resolutionMonthsLow+a.resolutionMonthsHigh)/2,lienPaid:lienPaid>=a.lien*(a.lienFractionLow+a.lienFractionHigh)/2};
    for(const {key} of focusDefinitions){
      const cell=accumulators.get(key);
      if(statuses[key]){cell.yesCount++;cell.yesSum+=presentValue;}
      else{cell.noCount++;cell.noSum+=presentValue;}
    }
  }
  results.sort();
  const mean=results.reduce((sum,value)=>sum+value,0)/iterations;
  const positive=results.filter(value=>value>0);
  const continuous=new Set(['treatmentContinuity','specialsSupport','comparativeFault','wageFraction','resolutionMonths','lienPaid']);
  const focus=focusDefinitions.map(definition=>{
    const cell=accumulators.get(definition.key),p=cell.yesCount/iterations;
    const withFactor=cell.yesCount?cell.yesSum/cell.yesCount:0,withoutFactor=cell.noCount?cell.noSum/cell.noCount:0;
    const swing=cell.yesCount&&cell.noCount?Math.abs(withFactor-withoutFactor):0;
    return {...definition,comparison:continuous.has(definition.key)?'above vs below reference':'present vs absent',probability:p,meanWithFactor:withFactor,meanWithoutFactor:withoutFactor,conditionalSwing:swing,
      focusValue:swing*4*p*(1-p)};
  }).sort((x,y)=>y.focusValue-x.focusValue);
  const maxFocus=focus[0]?.focusValue||1;
  for(const item of focus)item.priorityScore=Math.round(item.focusValue/maxFocus*100);
  return {
    outcomes:results,iterations,mean,liabilityRate:liableCount/iterations,zeroRate:zeroCount/iterations,
    policyLimitRate:policyLimitCount/iterations,pLiability:probabilities.liability,pScope:probabilities.scope,
    pSpecials:probabilities.specialsSupport,pWages:probabilities.wageSupport,pSurgery:probabilities.surgery,
    probabilities,nodeStats,nodeEstimates:summarizeNodeStats(nodeStats),
    gateDependence:dependence,p10:percentile(results,.10),p50:percentile(results,.50),p90:percentile(results,.90),
    conditionalP50:percentile(positive,.50),focus,
  };
}
