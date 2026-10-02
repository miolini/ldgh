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

const focusDefinitions=[
  {key:'liability',dimension:'liability',label:'Collision mechanism',action:'Obtain and reconcile scene photographs, independent incident accounts, and any police or transit records.'},
  {key:'scope',dimension:'scope',label:'Employment scope',action:'Secure vehicle assignment, dispatch, route, and employer records for the incident window.'},
  {key:'medical',dimension:'specialsSupport',label:'Past medical support',action:'Reconcile itemized provider ledgers, write-downs, payments, and no-fault allocations.'},
  {key:'wages',dimension:'wageSupport',label:'Wage-loss support',action:'Obtain post-2022 tax returns, commission statements, payroll records, and employer verification.'},
  {key:'surgery',dimension:'surgery',label:'Future surgery',action:'Confirm the current recommendation, medical necessity, timing, and cost range with the treating provider.'},
  {key:'comparativeFault',dimension:null,label:'Comparative fault',action:'Resolve vehicle positions, contact points, sight lines, and competing accounts before fixing the fault range.'},
];

export function simulateCase(assumptions,facts,iterations=1_000_000,rng=Math.random){
  const a=assumptions;
  const probabilities={
    liability:posterior(a.liability,facts,'liability'),scope:posterior(a.scope,facts,'scope'),
    specialsSupport:posterior(a.specialsSupport,facts,'specialsSupport'),wageSupport:posterior(a.wageSupport,facts,'wageSupport'),
    surgery:posterior(a.surgery,facts,'surgery'),
  };
  const results=new Float64Array(iterations);
  const accumulators=new Map(focusDefinitions.map(item=>[item.key,{yesCount:0,noCount:0,yesSum:0,noSum:0}]));
  let liableCount=0,zeroCount=0,policyLimitCount=0;
  const dependence=clamp(a.liabilityScopeDependence??0,0,.8);
  for(let i=0;i<iterations;i++){
    const mechanismProbability=beta(probabilities.liability,22,rng),scopeProbability=beta(probabilities.scope,22,rng);
    const sharedRank=rng()<dependence,commonRank=rng();
    const mechanismPass=(sharedRank?commonRank:rng())<mechanismProbability;
    const scopePass=(sharedRank?commonRank:rng())<scopeProbability;
    const liable=mechanismPass&&scopePass;
    const supportFraction=beta(probabilities.specialsSupport,30,rng);
    const pastMedical=Math.max(0,a.specials*supportFraction-a.noFaultPaid);
    const wagePass=rng()<beta(probabilities.wageSupport,22,rng);
    const supportedWages=wagePass?a.wages*triangular(.55,.82,1,rng):0;
    const surgeryPass=rng()<beta(probabilities.surgery,22,rng);
    const futureCare=surgeryPass?triangular(a.futureLow,(a.futureLow+a.futureHigh)/2,a.futureHigh,rng):0;
    const noneconomic=(pastMedical+futureCare)*triangular(a.nonEconomicLow,a.nonEconomicMode,a.nonEconomicHigh,rng);
    const comparative=triangular(a.comparativeLow,(a.comparativeLow+a.comparativeHigh)/2,a.comparativeHigh,rng);
    const gross=liable?(pastMedical+supportedWages+futureCare+noneconomic)*(1-comparative)*triangular(Math.max(.1,a.settlement-.1),a.settlement,Math.min(1,a.settlement+.1),rng):0;
    const policyHit=liable&&a.coverageLimit>0&&gross>=a.coverageLimit;
    const capped=a.coverageLimit>0?Math.min(gross,a.coverageLimit):gross;
    const net=Math.max(0,capped-a.lien);
    results[i]=net;
    liableCount+=liable?1:0;zeroCount+=net===0?1:0;policyLimitCount+=policyHit?1:0;
    const statuses={liability:mechanismPass,scope:scopePass,medical:supportFraction>=probabilities.specialsSupport,wages:wagePass,surgery:surgeryPass,comparativeFault:comparative>=((a.comparativeLow+a.comparativeHigh)/2)};
    for(const {key} of focusDefinitions){
      const cell=accumulators.get(key);
      if(statuses[key]){cell.yesCount++;cell.yesSum+=net;}
      else{cell.noCount++;cell.noSum+=net;}
    }
  }
  results.sort();
  const mean=results.reduce((sum,value)=>sum+value,0)/iterations;
  const positive=results.filter(value=>value>0);
  const focus=focusDefinitions.map(definition=>{
    const cell=accumulators.get(definition.key);
    const p=cell.yesCount/iterations;
    const withFactor=cell.yesCount?cell.yesSum/cell.yesCount:0;
    const withoutFactor=cell.noCount?cell.noSum/cell.noCount:0;
    const swing=Math.abs(withFactor-withoutFactor);
    return {...definition,probability:p,meanWithFactor:withFactor,meanWithoutFactor:withoutFactor,conditionalSwing:swing,
      focusValue:swing*4*p*(1-p)};
  }).sort((x,y)=>y.focusValue-x.focusValue);
  const maxFocus=focus[0]?.focusValue||1;
  for(const item of focus)item.priorityScore=Math.round(item.focusValue/maxFocus*100);
  return {
    outcomes:results,iterations,mean,liabilityRate:liableCount/iterations,zeroRate:zeroCount/iterations,
    policyLimitRate:policyLimitCount/iterations,pLiability:probabilities.liability,pScope:probabilities.scope,
    pSpecials:probabilities.specialsSupport,pWages:probabilities.wageSupport,pSurgery:probabilities.surgery,
    gateDependence:dependence,p10:percentile(results,.10),p50:percentile(results,.50),p90:percentile(results,.90),
    conditionalP50:percentile(positive,.50),focus,
  };
}
