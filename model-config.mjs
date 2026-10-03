// Scenario parameters are analyst judgments, not calibrated outcome estimates.
export const probabilityFactors = [
  {key:'liability',label:'Collision mechanism',group:'Responsibility',action:'Reconcile scene photographs, independent accounts and incident records.'},
  {key:'scope',label:'Employment scope',group:'Responsibility',action:'Obtain vehicle assignment, dispatch and employer records for the incident window.'},
  {key:'medicalCausation',label:'Injury causation',group:'Clinical evidence',action:'Compare pre-incident records, initial findings and treating opinions by body part.'},
  {key:'treatmentContinuity',label:'Treatment continuity',group:'Clinical evidence',action:'Reconcile treatment dates, discharge notes, gaps and subsequent referrals.'},
  {key:'specialsSupport',label:'Medical ledger support',group:'Clinical evidence',action:'Reconcile itemized charges, payments, write-downs and no-fault allocations.'},
  {key:'wageSupport',label:'Income documentation',group:'Income loss',action:'Obtain later tax returns, commission statements and employer verification.'},
  {key:'wageAttribution',label:'Work-loss attribution',group:'Income loss',action:'Link work restrictions and lost commissions to injury dates and alternative causes.'},
  {key:'surgeryNecessity',label:'Surgery necessity',group:'Future care',action:'Obtain a current treating opinion on indication, alternatives and incident-relatedness.'},
  {key:'surgery',label:'Surgery completion',group:'Future care',action:'Confirm scheduling, willingness, practical barriers and the provider cost estimate.'},
  {key:'lastingImpairment',label:'Lasting impairment',group:'Future care',action:'Obtain current functional findings, prognosis and any maximum-improvement assessment.'},
  {key:'collectability',label:'Collection success',group:'Recovery',action:'Confirm coverage availability, exclusions, other claimants and payment constraints.'},
];
export const dimensions=Object.fromEntries([...probabilityFactors.map(f=>[f.key,f.label]),['integrity','Integrity review only']]);
export const defaultAssumptions={
  liability:.62,scope:.62,liabilityScopeDependence:0,specialsSupport:.72,wageSupport:.38,surgery:.55,
  medicalCausation:.8,treatmentContinuity:.75,wageAttribution:.65,surgeryNecessity:.75,lastingImpairment:.6,collectability:.95,
  probabilityConcentration:22,medicalConcentration:30,
  specials:118400,wages:214000,noFaultPaid:50000,coverageLimit:100000,lien:22180,
  wageFractionLow:.55,wageFractionMode:.82,wageFractionHigh:1,
  futureLow:18000,futureHigh:65000,nonEconomicLow:.7,nonEconomicMode:1.45,nonEconomicHigh:2.8,
  transientMultiplier:.55,comparativeLow:.05,comparativeHigh:.35,
  settlementLow:.62,settlement:.72,settlementHigh:.82,
  lienFractionLow:1,lienFractionHigh:1,resolutionMonthsLow:6,resolutionMonthsHigh:30,annualDiscount:0,
};
const pct=v=>`${Math.round(v*100)}%`,cash=v=>`$${Math.round(v).toLocaleString('en-US')}`;
const range=(key,label,group,min,max,step,format=pct)=>({key,label,group,type:'range',min,max,step,format});
const amount=(key,label,group,max=1_000_000,step=1000)=>({key,label,group,type:'number',min:0,max,step,format:cash});
export const assumptionControls=[
  ...probabilityFactors.map(f=>range(f.key,`${f.label} · starting estimate`,f.group,.01,.99,.01)),
  range('liabilityScopeDependence','Shared responsibility draw','Responsibility',0,.8,.05),
  range('comparativeLow','Comparative fault · low','Responsibility',0,.95,.01),
  range('comparativeHigh','Comparative fault · high','Responsibility',0,.95,.01),
  amount('specials','Medical charges claimed','Clinical evidence'),amount('noFaultPaid','No-fault excluded in this scenario','Clinical evidence',500000),
  amount('wages','Wage loss claimed','Income loss'),
  ...['Low','Mode','High'].map(s=>range(`wageFraction${s}`,`Supported income fraction · ${s.toLowerCase()}`,'Income loss',0,1,.01)),
  amount('futureLow','Future care cost · low','Future care',500000),amount('futureHigh','Future care cost · high','Future care',500000),
  ...['Low','Mode','High'].map(s=>range(`nonEconomic${s}`,`Non-economic multiplier · ${s.toLowerCase()}`,'Future care',0,6,.05,v=>`${v.toFixed(2)}×`)),
  range('transientMultiplier','Multiplier if no lasting impairment','Future care',0,1,.01),
  amount('coverageLimit','Coverage cap · 0 = uncapped','Recovery',5000000,5000),amount('lien','Recorded lien','Recovery',500000,500),
  range('lienFractionLow','Lien paid fraction · low','Recovery',0,1,.01),range('lienFractionHigh','Lien paid fraction · high','Recovery',0,1,.01),
  range('settlementLow','Resolution factor · low','Recovery',0,1,.01),range('settlement','Resolution factor · mode','Recovery',0,1,.01),range('settlementHigh','Resolution factor · high','Recovery',0,1,.01),
  ...['Low','High'].map(s=>({key:`resolutionMonths${s}`,label:`Resolution months · ${s.toLowerCase()}`,group:'Recovery',type:'number',min:0,max:120,step:1,format:v=>`${v} months`})),
  range('annualDiscount','Annual time discount · optional','Recovery',0,.3,.01),
  ...['probability','medical'].map(s=>({key:`${s}Concentration`,label:`${s==='medical'?'Continuous support':'Event probability'} concentration`,group:'Uncertainty',type:'number',min:2,max:200,step:1,format:v=>`${v} · higher = narrower`})),
];
export const orderedRanges=[['futureLow','futureHigh'],['comparativeLow','comparativeHigh'],['wageFractionLow','wageFractionMode','wageFractionHigh'],['nonEconomicLow','nonEconomicMode','nonEconomicHigh'],['settlementLow','settlement','settlementHigh'],['lienFractionLow','lienFractionHigh'],['resolutionMonthsLow','resolutionMonthsHigh']];
export function normalizeAssumptions(input={}){
  const output={...defaultAssumptions};
  // Older saved scenarios only exposed the resolution mode; preserve it on migration.
  if(Number.isFinite(Number(input.settlement))&&input.settlement!=null&&input.settlementLow==null&&input.settlementHigh==null){
    output.settlementLow=Math.max(0,Number(input.settlement)-.1);output.settlementHigh=Math.min(1,Number(input.settlement)+.1);
  }
  for(const c of assumptionControls){const v=Number(input[c.key]);if(Number.isFinite(v))output[c.key]=Math.min(c.max,Math.max(c.min,v));}
  for(const keys of orderedRanges){const values=keys.map(k=>output[k]).sort((a,b)=>a-b);keys.forEach((k,i)=>output[k]=values[i]);}
  return output;
}
export function validateAssumptionUpdate(base,key,value){
  const c=assumptionControls.find(c=>c.key===key),v=Number(value);
  if(!c||!Number.isFinite(v)||v<c.min||v>c.max)throw new Error(`Invalid ${key} value${c?`; allowed range ${c.min}–${c.max}`:''}.`);
  const next={...base,[key]:v};
  for(const keys of orderedRanges)if(keys.some((k,i)=>i>0&&next[keys[i-1]]>next[k]))throw new Error(`Keep ${keys.join(' ≤ ')} in order.`);
  return next;
}
