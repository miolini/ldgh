import { probabilityFactors, dimensions } from './model-config.mjs?build=20261002-expanded-graph';
import { evidenceLikelihoodRatio } from './probabilistic-model.mjs?build=20261002-expanded-graph';
const pct=v=>`${(v*100).toFixed(1)}%`,money=v=>`$${Math.round(v).toLocaleString('en-US')}`;
export function buildFactorGraph(a,facts,result){
  const nodes=[],edges=[];
  const add=(id,x,y,kind,title,metric,caption,copy,items=[],key=null)=>{
    const n={id,x,y,w:x===1380?265:235,h:116,kind,title,metric,caption,copy,items,key};nodes.push(n);return n;
  };
  const link=(from,to)=>edges.push({from,to});
  const est=(key,x,y,title,format=money,copy='',items=[])=>{
    const s=result.nodeEstimates[key];
    return add(key,x,y,'model',title,`Mean ${format(s.mean)}`,`10–90% draws ${format(s.p10)}–${format(s.p90)}`,copy,
      [...items,`Sampled mean: ${format(s.mean)}`,`Approximate 10th–90th percentile: ${format(s.p10)}–${format(s.p90)}`,`${result.iterations.toLocaleString('en-US')} cumulative trials`],({pastMedical:'specials',supportedWages:'wages',futureCare:'futureLow',noneconomic:'nonEconomicMode',damages:'specials',gross:'settlement',covered:'coverageLimit',collected:'collectability',net:'lien',presentValue:'annualDiscount',comparativeFault:'comparativeHigh',resolutionFactor:'settlement',resolutionMonths:'resolutionMonthsHigh',lienPaid:'lien'})[key]||key);
  };
  const evidence=(id,y,keys)=>{
    const matching=facts.filter(f=>keys.includes(f.dimension));
    add(`${id}-evidence`,15,y,'evidence',id,`${matching.length} source observations`,'Select to inspect weights and sources',
      'Facts shift starting odds using analyst-assigned likelihood ratios and source confidence. Same-source contributions are partially discounted; different sources may still be dependent.',
      matching.length?matching.map(f=>`${f.text} · ${dimensions[f.dimension]} · LR ${f.lr} · ${Math.round(f.confidence*100)}% confidence · ${f.source||'No source reference'}`):['No assigned observations. The starting estimates remain illustrative.']);
  };
  const probability=(key,x,y,evidenceId)=>{
    const f=probabilityFactors.find(f=>f.key===key),s=result.nodeEstimates[key],p=result.probabilities[key];
    const concentration=['specialsSupport','treatmentContinuity'].includes(key)?a.medicalConcentration:a.probabilityConcentration;
    add(`${key}-posterior`,x,y,'probability',f.label,pct(p),`Probability draws ${pct(s.p10)}–${pct(s.p90)}`,
      `Evidence-adjusted mean of a Beta distribution with concentration ${concentration}. This parameter is analyst judgment; increasing concentration narrows assumed uncertainty and does not add evidence.`,
      [`Starting estimate: ${pct(a[key])}`,`Weighted evidence LR: ${evidenceLikelihoodRatio(facts,key).toFixed(2)}×`,s.eventRate==null?'Continuous support fraction, not a pass/fail event':`Realized event frequency: ${pct(s.eventRate)}`,`Investigation: ${f.action}`],key);
    link(`${evidenceId}-evidence`,`${key}-posterior`);
  };
  const rows=[
    ['Responsibility',65,['liability','scope']],['Clinical linkage',225,['medicalCausation','treatmentContinuity']],
    ['Medical costs',385,['specialsSupport']],['Income loss',545,['wageSupport','wageAttribution']],
    ['Future care',705,['surgeryNecessity','surgery']],['Prognosis',865,['lastingImpairment']],['Collection',1025,['collectability']],
  ];
  for(const [title,y,keys] of rows){evidence(title,y,keys);keys.forEach((key,i)=>probability(key,285+i*270,y,title));}
  add('liability-gate',825,65,'model','Responsibility gate',pct(result.liabilityRate),'Mechanism AND employment scope',
    'This scenario requires both responsibility events to pass. Shared-percentile sampling is a dependence assumption, not an established legal finding.',[`Shared draw rate: ${pct(a.liabilityScopeDependence)}`]);
  link('liability-posterior','liability-gate');link('scope-posterior','liability-gate');
  add('injury-link',825,225,'model','Shared injury link',pct(result.nodeEstimates.medicalCausation.eventRate),'One event links three loss branches',
    'A single causation event is shared by medical costs, income loss and future care. Treatment continuity samples medical support and contributes to the lasting-impairment scenario.',
    ['Prior injury discrepancies require body-part-specific reconciliation. They are not themselves proof of fraud.']);
  link('medicalCausation-posterior','injury-link');
  est('pastMedical',825,385,'Supported medical',money,'If injury causation passes: charges × ledger support × continuity, less the configured no-fault exclusion. This offset is a scenario assumption requiring ledger reconciliation.',[`Claimed charges: ${money(a.specials)}`,`No-fault excluded: ${money(a.noFaultPaid)}`]);
  link('specialsSupport-posterior','pastMedical');link('injury-link','pastMedical');link('treatmentContinuity-posterior','pastMedical');
  est('supportedWages',825,545,'Supported income loss',money,'Requires the shared causation event, documented income and work-loss attribution. The accepted amount then uses the configured triangular fraction range.',[`Claim: ${money(a.wages)}`,`Supported fraction: ${pct(a.wageFractionLow)} / ${pct(a.wageFractionMode)} / ${pct(a.wageFractionHigh)}`]);
  link('wageSupport-posterior','supportedWages');link('wageAttribution-posterior','supportedWages');link('injury-link','supportedWages');
  est('futureCare',825,705,'Future care cost',money,'Requires incident-related injury, surgery necessity and completion. Cost is triangular between the configured bounds.',[`Cost: ${money(a.futureLow)}–${money(a.futureHigh)}`,`Combined occurrence: ${(result.probabilities.medicalCausation*result.probabilities.surgeryNecessity*result.probabilities.surgery*100).toFixed(1)}% under independent draws`]);
  link('surgeryNecessity-posterior','futureCare');link('surgery-posterior','futureCare');link('injury-link','futureCare');
  est('noneconomic',825,865,'Non-economic scenario',money,'Supported medical plus future care times a sampled multiplier. Without causation, lasting impairment and continuity events, the multiplier is reduced by the transient-impairment factor. This is a valuation proxy, not an injury-specific damages formula.',[`Multiplier: ${a.nonEconomicLow} / ${a.nonEconomicMode} / ${a.nonEconomicHigh}×`,`Transient multiplier factor: ${pct(a.transientMultiplier)}`]);
  link('lastingImpairment-posterior','noneconomic');link('treatmentContinuity-posterior','noneconomic');link('injury-link','noneconomic');link('pastMedical','noneconomic');link('futureCare','noneconomic');
  est('lienPaid',555,1025,'Lien deduction',money,'Recorded lien times a triangular payment fraction. Fixed at 100% by default; reduce only with a supported scenario.',[`Recorded lien: ${money(a.lien)}`,`Paid fraction: ${pct(a.lienFractionLow)}–${pct(a.lienFractionHigh)}`]);
  est('collected',825,1025,'Collected recovery',money,'Covered recovery is payable only when the collection event passes. This represents a payment scenario, not a determination of insurance obligations.');
  link('collectability-posterior','collected');
  est('comparativeFault',285,1185,'Comparative fault',pct,'Triangular scenario range; no evidence-to-fault percentage calibration is assumed.',[`Range: ${pct(a.comparativeLow)}–${pct(a.comparativeHigh)}`]);
  est('resolutionFactor',555,1185,'Resolution factor',pct,'Triangular proportion applied after supported damages and comparative fault. This is an editable resolution scenario.',[`Low / mode / high: ${pct(a.settlementLow)} / ${pct(a.settlement)} / ${pct(a.settlementHigh)}`]);
  est('resolutionMonths',825,1185,'Resolution time',v=>`${v.toFixed(1)} mo`,'Triangular months to resolution. The optional annual discount affects present value only.',[`Range: ${a.resolutionMonthsLow}–${a.resolutionMonthsHigh} months`,`Annual discount: ${pct(a.annualDiscount)}`]);
  est('damages',1110,465,'Supported damages',money,'Sum of supported medical, income loss, future care and non-economic scenario value before responsibility, comparative fault and resolution discounts.');
  for(const key of ['pastMedical','supportedWages','futureCare','noneconomic'])link(key,'damages');
  est('gross',1110,685,'Gross resolution',money,'Responsibility gate × supported damages × (1 − comparative fault) × resolution factor.');
  for(const key of ['damages','liability-gate','comparativeFault','resolutionFactor'])link(key,'gross');
  est('covered',1110,905,'Covered amount',money,'Configured coverage cap applied to gross resolution. Zero cap means uncapped; additional coverage is not inferred.',[`Configured cap: ${a.coverageLimit?money(a.coverageLimit):'Uncapped'}`,`Cap reached: ${pct(result.policyLimitRate)} of trials`]);
  link('gross','covered');link('covered','collected');
  est('net',1380,1025,'Net after lien',money,'Maximum of zero and collected recovery minus lien deduction. Attorney fees and case expenses are not included.');
  link('collected','net');link('lienPaid','net');
  est('presentValue',1380,1185,'Time-adjusted recovery',money,'Net recovery divided by (1 + annual discount)^(months / 12). With the default 0% discount, this equals net recovery.',[`Exact median: ${money(result.p50)}`,`Exact 80% outcome range: ${money(result.p10)}–${money(result.p90)}`,`Zero recovery: ${pct(result.zeroRate)}`]);
  link('net','presentValue');link('resolutionMonths','presentValue');
  return {nodes,edges,width:1690,height:1340};
}
