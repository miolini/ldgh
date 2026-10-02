import { evidenceLikelihoodRatio } from './probabilistic-model.mjs';

const STORAGE_KEY = 'fieldnote-sapini-analysis-v2';
const number = new Intl.NumberFormat('en-US', {maximumFractionDigits:0});
const money = value => `$${number.format(Math.max(0, value))}`;
const clamp = (n, min, max) => Math.min(max, Math.max(min, Number(n)));
const dimensions = {
  liability: 'Liability', specialsSupport: 'Past medical support',
  scope:'Employment scope', wageSupport: 'Wage support', surgery: 'Future surgery', integrity: 'Integrity review only',
};

const defaults = {
  assumptions: {
    liability: .62, scope: .62, liabilityScopeDependence:0, specialsSupport: .72, wageSupport: .38, surgery: .55,
    specials: 118400, wages: 214000, futureLow: 18000, futureHigh: 65000,
    nonEconomicLow: .7, nonEconomicMode: 1.45, nonEconomicHigh: 2.8,
    comparativeLow: .05, comparativeHigh: .35, settlement: .72, coverageLimit:100000, lien:22180, noFaultPaid:50000,
  },
  facts: [
    {id:'seed-accounts', text:'Incident accounts differ across the file', source:'Liability note · May 23, 2023', dimension:'liability', lr:.72, confidence:.55, seeded:true},
    {id:'seed-police', text:'No police accident report is recorded', source:'Matter snapshot · source inventory', dimension:'liability', lr:.88, confidence:.35, seeded:true},
    {id:'seed-scope', text:'Vehicle assignment and dispatch records remain outstanding', source:'Discovery note · Sep 6, 2026', dimension:'scope', lr:.58, confidence:.75, seeded:true},
    {id:'seed-surgery', text:'Recommended right shoulder surgery remains undated', source:'Orthopaedic follow-up · Sep 24, 2026', dimension:'surgery', lr:.72, confidence:.7, seeded:true},
    {id:'seed-wage', text:'$214,000 wage claim currently rests on a 2022 1099', source:'Matter field · wage-loss task', dimension:'wageSupport', lr:.52, confidence:.8, seeded:true},
    {id:'seed-ledger', text:'Large medical charges await line-item reconciliation', source:'Open task · Oct 5, 2026', dimension:'specialsSupport', lr:.66, confidence:.7, seeded:true},
    {id:'seed-coverage', text:'Adjuster confirmed $100,000 per-person limit', source:'Coverage confirmation · Sep 8, 2026', dimension:'integrity', lr:1, confidence:.9, seeded:true},
  ],
};

function loadState() {
  try {
    const stored = JSON.parse(localStorage.getItem(STORAGE_KEY));
    if (!stored || !stored.assumptions || !Array.isArray(stored.facts)) return structuredClone(defaults);
    return {
      assumptions: {...defaults.assumptions, ...stored.assumptions},
      facts: stored.facts.map(fact => ({...fact, lr:clamp(fact.lr, .25, 4), confidence:clamp(fact.confidence, 0, 1)})),
    };
  } catch { return structuredClone(defaults); }
}

const state = loadState();
let lastResult;
let editingFactId = null;
let navigateWorkspace;
const $ = selector => document.querySelector(selector);

function save() {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(state)); }
  catch (error) { console.warn('Analyst inputs could not be persisted:', error); }
}

function percentile(sorted, fraction) { return sorted[Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * fraction))] ?? 0; }

function renderChart(outcomes) {
  const svg = $('#outcome-chart');
  if (!svg) return;
  const width=900,height=310,left=38,right=12,top=12,bottom=26,plotW=width-left-right,plotH=height-top-bottom,bins=52;
  const ceiling=Math.max(10000,percentile(outcomes,.98));
  const counts=new Array(bins).fill(0);
  const chartWeight=lastResult.iterations/outcomes.length;
  for (const value of outcomes) counts[Math.min(bins-1,Math.floor(value/ceiling*bins))]+=chartWeight;
  const peak=Math.max(...counts,1);
  const barW=plotW/bins;
  const bars=counts.map((count,index)=>{
    const h=count/peak*plotH;
    const x=left+index*barW;
    return `<rect x="${x.toFixed(1)}" y="${(top+plotH-h).toFixed(1)}" width="${Math.max(1,barW-2).toFixed(1)}" height="${h.toFixed(1)}" rx="1" fill="#222" opacity="${index===0?'.35':'.88'}"><title>${money(index/bins*ceiling)}–${money((index+1)/bins*ceiling)} · ${number.format(Math.round(count))} scenarios</title></rect>`;
  }).join('');
  const marker=(value,label)=>{
    const x=left+Math.min(1,value/ceiling)*plotW;
    return `<line x1="${x}" x2="${x}" y1="${top}" y2="${top+plotH}" stroke="#666" stroke-dasharray="3 4"/><text x="${Math.min(width-52,Math.max(left+2,x+3))}" y="${top+10}" font-size="9" fill="#555" font-family="DM Mono,monospace">${label}</text>`;
  };
  const grid=[0,.25,.5,.75,1].map(v=>{
    const y=top+plotH*(1-v);
    return `<line x1="${left}" x2="${width-right}" y1="${y}" y2="${y}" stroke="#e8e8e8"/><text x="${left-7}" y="${y+3}" text-anchor="end" font-size="8" fill="#999" font-family="DM Mono,monospace">${Math.round(peak*v)}</text>`;
  }).join('');
  svg.innerHTML=`<title>${number.format(lastResult.iterations)} Monte Carlo net recovery scenarios</title>${grid}${bars}${marker(lastResult.p50,'P50')}${marker(lastResult.p90,'P90')}`;
  $('#chart-axis-mid').textContent=money(ceiling/2);
  $('#chart-axis-max').textContent=`${money(ceiling)}+`;
}

const controls = [
  {key:'liability',label:'Liability starting probability',type:'range',min:.1,max:.9,step:.01,format:v=>`${Math.round(v*100)}%`},
  {key:'scope',label:'Employment-scope starting probability',type:'range',min:.1,max:.95,step:.01,format:v=>`${Math.round(v*100)}%`},
  {key:'liabilityScopeDependence',label:'Chance liability and scope share a percentile draw',type:'range',min:0,max:.8,step:.05,format:v=>`${Math.round(v*100)}%`},
  {key:'specialsSupport',label:'Past medical support prior',type:'range',min:.1,max:.99,step:.01,format:v=>`${Math.round(v*100)}%`},
  {key:'wageSupport',label:'Wage support prior',type:'range',min:.05,max:.95,step:.01,format:v=>`${Math.round(v*100)}%`},
  {key:'surgery',label:'Future surgery scenario rate',type:'range',min:.05,max:.95,step:.01,format:v=>`${Math.round(v*100)}%`},
  {key:'specials',label:'Past medical specials claimed',type:'number',min:0,max:1000000,step:1000,format:money},
  {key:'wages',label:'Wage loss claimed',type:'number',min:0,max:1000000,step:1000,format:money},
  {key:'noFaultPaid',label:'No-fault paid/exhausted · excluded',type:'number',min:0,max:500000,step:1000,format:money},
  {key:'coverageLimit',label:'Per-person coverage limit · 0 = uncapped',type:'number',min:0,max:5000000,step:5000,format:v=>v===0?'Uncapped':money(v)},
  {key:'lien',label:'Recorded Medicaid lien',type:'number',min:0,max:500000,step:500,format:money},
  {key:'futureLow',label:'Future care · low',type:'number',min:0,max:500000,step:1000,format:money},
  {key:'futureHigh',label:'Future care · high',type:'number',min:0,max:500000,step:1000,format:money},
  {key:'nonEconomicMode',label:'Non-economic multiplier · mode',type:'range',min:.2,max:5,step:.05,format:v=>`${v.toFixed(2)}×`},
  {key:'comparativeHigh',label:'Comparative fault · high',type:'range',min:0,max:.8,step:.01,format:v=>`${Math.round(v*100)}%`},
  {key:'settlement',label:'Resolution factor · mode',type:'range',min:.25,max:1,step:.01,format:v=>`${Math.round(v*100)}%`},
];

function renderAssumptions() {
  const container=$('#assumptions-grid');
  if(!container) return;
  container.replaceChildren(...controls.map(control=>{
    const wrapper=document.createElement('div');wrapper.className='assumption-control';
    const label=document.createElement('label');label.htmlFor=`assumption-${control.key}`;label.textContent=control.label;
    const output=document.createElement('output');output.htmlFor=`assumption-${control.key}`;output.textContent=control.format(state.assumptions[control.key]);
    const input=document.createElement('input');input.id=`assumption-${control.key}`;input.type=control.type;input.min=control.min;input.max=control.max;input.step=control.step;input.value=state.assumptions[control.key];
    input.addEventListener('input',()=>{
      state.assumptions[control.key]=clamp(input.value,control.min,control.max);
      output.textContent=control.format(state.assumptions[control.key]);
      save(); scheduleRun();
    });
    label.append(output);wrapper.append(label,input);return wrapper;
  }));
}

function renderFacts() {
  const list=$('#model-fact-list');
  const count=$('#fact-count');
  if(count) count.textContent=String(state.facts.length);
  renderAnalystFacts();
  if(!list) return;
  list.replaceChildren(...state.facts.map(fact=>{
    const row=document.createElement('div');row.className='model-fact-row';
    const summary=document.createElement('div');
    const title=document.createElement('strong');title.textContent=fact.text;
    const meta=document.createElement('small');meta.textContent=`${dimensions[fact.dimension]??fact.dimension} · ${fact.source||'Analyst supplied'}`;
    summary.append(title,meta);
    const edit=document.createElement('div');edit.className='fact-edit-controls';
    const lr=document.createElement('input');lr.type='number';lr.min='.25';lr.max='4';lr.step='.05';lr.value=fact.lr;lr.title='Likelihood ratio: 1 is neutral';lr.setAttribute('aria-label',`Likelihood ratio for ${fact.text}`);
    const confidence=document.createElement('input');confidence.type='range';confidence.min='0';confidence.max='1';confidence.step='.05';confidence.value=fact.confidence;confidence.title='Source confidence';confidence.setAttribute('aria-label',`Source confidence for ${fact.text}`);
    const del=document.createElement('button');del.type='button';del.textContent='×';del.title='Remove analyst fact';del.setAttribute('aria-label',`Remove ${fact.text}`);
    lr.addEventListener('change',()=>{fact.lr=clamp(lr.value,.25,4);lr.value=fact.lr;save();renderAnalystFacts();renderEvidence(records);run();});
    confidence.addEventListener('input',()=>{fact.confidence=clamp(confidence.value,0,1);save();renderAnalystFacts();scheduleRun();});
    del.addEventListener('click',()=>{state.facts=state.facts.filter(item=>item.id!==fact.id);save();renderFacts();renderEvidence(records);run();});
    edit.append(lr,confidence,del);row.append(summary,edit);return row;
  }));
}

function openFactEditor(fact) {
  editingFactId=fact.id;
  const form=$('#fact-form');
  if(!form) return;
  form.elements.text.value=fact.text;
  form.elements.source.value=fact.source||'';
  form.elements.dimension.value=fact.dimension;
  form.elements.lr.value=fact.lr;
  form.elements.confidence.value=fact.confidence;
  $('#fact-confidence-label').textContent=`${Math.round(fact.confidence*100)}%`;
  $('#fact-form-title').textContent='Edit investigator fact';
  form.querySelector('[type="submit"]').textContent='Save fact changes';
  $('#fact-cancel').hidden=false;
  navigateWorkspace?.('facts');
  requestAnimationFrame(()=>{$('#fact-entry').scrollIntoView({behavior:'smooth',block:'start'});form.elements.text.focus();});
}

function renderAnalystFacts() {
  const list=$('#analyst-fact-list');
  const count=$('#fact-list-count');
  if(count) count.textContent=String(state.facts.length);
  if(!list) return;
  list.replaceChildren(...state.facts.map(fact=>{
    const row=document.createElement('article');row.className='analyst-fact-row';
    const text=document.createElement('strong');text.textContent=fact.text;
    const meta=document.createElement('p');meta.textContent=`${dimensions[fact.dimension]??fact.dimension} · LR ${Number(fact.lr).toFixed(2)} · ${Math.round(fact.confidence*100)}% confidence · ${fact.source||'Analyst supplied'}`;
    const edit=document.createElement('button');edit.type='button';edit.className='quiet-button';edit.textContent='Edit fact';edit.addEventListener('click',()=>openFactEditor(fact));
    row.append(text,meta,edit);return row;
  }));
}

function renderAnomalies() {
  const signals=[
    {title:'Collision mechanism needs corroboration',detail:'The snapshot records three incident accounts. The 22 scene photographs described at intake have not been obtained; no police report is on file.',source:'Liability note · May 23, 2023',needle:'three different accounts'},
    {title:'Employment scope is not independently established',detail:'The vehicle assignment and dispatch records identified as central to scope have not been produced.',source:'Discovery note · Sep 6, 2026',needle:'vehicle assignment and dispatch'},
    {title:'Prior injury history needs reconciliation',detail:'The intake answer conflicts with older ankle imaging in the client document stack; confirm body part, dates, and disclosure history.',source:'Matter note · May 31, 2023',needle:'prior injury discrepancy'},
    {title:'Wage loss has limited year coverage',detail:'The $214,000 claim references a 2022 1099; later commission records remain requested.',source:'Open task · Sep 26, 2026',needle:'commission records'},
    {title:'Medical ledger reconciliation remains open',detail:'The current specials figure includes charges whose itemized lines are still being reconciled.',source:'Open task · Oct 5, 2026',needle:'reconcile chiropractic and pt ledgers'},
    {title:'Treatment chronology requires audit',detail:'The physical therapy file notes treatment entries continuing after a discharge note; verify the source timeline before drawing conclusions.',source:'Records note · Nov 19, 2023',needle:'discharge note is followed'},
  ];
  for(const fact of state.facts.filter(item=>!item.seeded&&item.dimension==='integrity')) signals.push({title:'Analyst-entered integrity review',detail:fact.text,source:fact.source||'Analyst supplied'});
  const list=$('#anomaly-list');$('#anomaly-count').textContent=String(signals.length);
  list.replaceChildren(...signals.map(signal=>{
    const row=document.createElement('article');row.className='anomaly-row';
    const title=document.createElement('strong');title.textContent=signal.title;
    const detail=document.createElement('p');detail.textContent=`${signal.detail} · ${signal.source}`;row.append(title,detail);
    const target=(window.caseRecords||[]).find(item=>`${item.title} ${item.description}`.toLowerCase().includes(signal.needle||''));
    if(target){const open=document.createElement('button');open.type='button';open.textContent='Open source record ↗';open.addEventListener('click',()=>window.dispatchEvent(new CustomEvent('case:open-record',{detail:target})));row.append(open);}
    return row;
  }));
}

let selectedGraphNode='liability-posterior';
function renderFactorGraph(result) {
  const svg=$('#factor-graph');
  if(!svg)return;
  const pct=value=>`${(value*100).toFixed(1)}%`;
  const escapeSvg=value=>String(value).replace(/[&<>"']/g,char=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&apos;'}[char]));
  const groups=[
    {id:'liability',label:'Mechanism / liability',p:result.pLiability,y:38},
    {id:'scope',label:'Employment scope',p:result.pScope,y:153},
    {id:'specialsSupport',label:'Past medical support',p:result.pSpecials,y:268},
    {id:'wageSupport',label:'Wage support',p:result.pWages,y:383},
    {id:'surgery',label:'Future surgery',p:result.pSurgery,y:498},
  ];
  const descriptors={},nodes=[],edges=[];
  const factsFor=dimension=>state.facts.filter(fact=>fact.dimension===dimension);
  const weightedLR=dimension=>evidenceLikelihoodRatio(state.facts,dimension);
  function node(id,x,y,w,h,kind,title,metric,caption) {
    descriptors[id]={kind,title,metric,caption,items:[]};
    nodes.push(`<g class="factor-node factor-node-${kind}" data-node="${escapeSvg(id)}" tabindex="0" role="button" aria-label="${escapeSvg(`${title}: ${metric}. Select for details.`)}"><rect class="factor-node-box" x="${x}" y="${y}" width="${w}" height="${h}" rx="6"/><text class="factor-node-kind" x="${x+13}" y="${y+18}">${kind==='evidence'?'ANALYST FACTS':kind==='probability'?'POSTERIOR':'SIMULATION'}</text><text class="factor-node-title" x="${x+13}" y="${y+41}">${escapeSvg(title)}</text><text class="factor-node-metric" x="${x+13}" y="${y+64}">${escapeSvg(metric)}</text><text class="factor-node-caption" x="${x+13}" y="${y+h-9}">${escapeSvg(caption)}</text></g>`);
  }
  const connect=(x1,y1,x2,y2,curve=0)=>edges.push(`<path class="factor-edge" d="M ${x1} ${y1} C ${x1+curve} ${y1}, ${x2-curve} ${y2}, ${x2} ${y2}" marker-end="url(#factor-arrow)"/>`);
  for(const group of groups){
    const facts=factsFor(group.id),lr=weightedLR(group.id),prior=state.assumptions[group.id];
    const evidenceId=`${group.id}-evidence`,posteriorId=`${group.id}-posterior`;
    node(evidenceId,20,group.y,245,82,'evidence',group.label,`${facts.length} fact${facts.length===1?'':'s'} · weighted LR ${lr.toFixed(2)}×`,'Analyst inputs · source confidence');
    node(posteriorId,315,group.y,235,82,'probability',group.label,pct(group.p),`Starting value ${pct(prior)} · evidence-adjusted`);
    descriptors[evidenceId].copy='Observed case facts update starting odds through confidence-weighted likelihood ratios. Same-source contributions are discounted by the square root of the number of facts from that source; weights are analyst judgments, not learned parameters.';
    descriptors[evidenceId].items=facts.length?facts.map(fact=>`${fact.text} — LR ${Number(fact.lr).toFixed(2)}× · ${Math.round(fact.confidence*100)}% confidence${fact.source?` · ${fact.source}`:''}`):['No analyst facts are currently assigned to this factor.'];
    descriptors[posteriorId].copy='This evidence-adjusted probability is used as the mean of a Beta distribution in each Monte Carlo trial.';
    descriptors[posteriorId].items=[`Starting probability: ${pct(prior)}`,`Combined confidence-weighted likelihood ratio: ${lr.toFixed(2)}×`,`Posterior probability: ${pct(group.p)}`,'Sampling distribution: Beta, concentration 22'];
    connect(265,group.y+41,315,group.y+41,18);
  }
  node('liability-gate',620,124,235,104,'model','Liability gate',pct(result.liabilityRate),'Both mechanism and scope must pass');
  descriptors['liability-gate'].copy='Each trial samples mechanism and employment scope. At the configured rate both gates use the same percentile draw; otherwise each uses a separate draw. Recovery proceeds only when both pass.';
  descriptors['liability-gate'].items=[`Mechanism posterior: ${pct(result.pLiability)}`,`Employment-scope posterior: ${pct(result.pScope)}`,`Simulated conjunction frequency: ${pct(result.liabilityRate)}`,`Shared percentile draw: ${pct(result.gateDependence)} · 0% means separate draws`];
  node('damages',620,326,235,104,'model','Supported damages','5 uncertain components','Medical · wages · surgery · fault · multiplier');
  descriptors['damages'].copy='Sampled support and dollar factors determine gross damages. Past medical charges are reduced by exhausted no-fault benefits.';
  descriptors['damages'].items=[`Medical support: ${pct(result.pSpecials)} · specials ${money(state.assumptions.specials)}`,`Wage support: ${pct(result.pWages)} · claim ${money(state.assumptions.wages)}`,`Future surgery: ${pct(result.pSurgery)} · range ${money(state.assumptions.futureLow)}–${money(state.assumptions.futureHigh)}`,`Noneconomic multiplier: ${state.assumptions.nonEconomicLow.toFixed(2)}–${state.assumptions.nonEconomicHigh.toFixed(2)}×`,`Comparative fault: ${(state.assumptions.comparativeLow*100).toFixed(0)}–${(state.assumptions.comparativeHigh*100).toFixed(0)}%`];
  node('constraints',620,520,235,92,'model','Recovery constraints',money(state.assumptions.coverageLimit),'Coverage cap · lien · settlement factor');
  descriptors['constraints'].copy='The scenario applies the settlement factor, caps gross recovery at the configured coverage limit, then subtracts the lien. Assumptions can be changed on the Assumptions page.';
  descriptors['constraints'].items=[`Per-person coverage limit: ${money(state.assumptions.coverageLimit)}`,`Medicaid lien: ${money(state.assumptions.lien)}`,`Exhausted no-fault excluded: ${money(state.assumptions.noFaultPaid)}`,`Settlement factor mode: ${pct(state.assumptions.settlement)}`];
  node('net-recovery',905,282,210,126,'model','Net recovery',money(result.p50),`80% range ${money(result.p10)}–${money(result.p90)}`);
  descriptors['net-recovery'].copy='The output is a distribution of modeled net recovery scenarios, not a case valuation or prediction. It includes zero-recovery outcomes.';
  descriptors['net-recovery'].items=[`Unconditional median: ${money(result.p50)}`,`Median among positive outcomes: ${money(result.conditionalP50)}`,`10th–90th percentile: ${money(result.p10)}–${money(result.p90)}`,`${number.format(result.iterations)} simulation trials`];
  connect(550,79,620,158,26);connect(550,194,620,194,16);
  connect(550,309,620,360,30);connect(550,424,620,378,30);connect(550,539,620,405,55);
  connect(855,176,905,315,36);connect(855,378,905,345,32);connect(855,566,905,375,65);
  svg.innerHTML=`<defs><marker id="factor-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#aaa"/></marker></defs><text class="factor-column-label" x="20" y="19">CASE EVIDENCE</text><text class="factor-column-label" x="315" y="19">BAYESIAN UPDATE</text><text class="factor-column-label" x="620" y="105">FACTOR INTERACTION</text><text class="factor-column-label" x="905" y="260">SIMULATED RESULT</text>${edges.join('')}${nodes.join('')}`;
  const select=id=>{
    const detail=descriptors[id];if(!detail)return;
    selectedGraphNode=id;
    svg.querySelectorAll('.factor-node').forEach(item=>item.classList.toggle('is-selected',item.dataset.node===id));
    $('#node-detail-type').textContent=detail.kind==='evidence'?'EVIDENCE INPUT':detail.kind==='probability'?'POSTERIOR PROBABILITY':'SCENARIO FACTOR';
    $('#node-detail-title').textContent=detail.title;$('#node-detail-metric').textContent=detail.metric;$('#node-detail-copy').textContent=detail.copy||detail.caption;
    $('#node-detail-list').replaceChildren(...detail.items.map(value=>{const li=document.createElement('li');li.textContent=value;return li;}));
  };
  svg.querySelectorAll('.factor-node').forEach(item=>{
    item.addEventListener('click',()=>select(item.dataset.node));
    item.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();select(item.dataset.node);}});
  });
  select(descriptors[selectedGraphNode]?selectedGraphNode:'liability-posterior');
}

const SIMULATION_ITERATIONS=1_000_000;
let cancelSimulation;
let activeRun=0;
function renderResult(result, complete=false) {
  lastResult=result;
  $('#result-median').textContent=money(result.p50);
  $('#result-conditional-median').textContent=money(result.conditionalP50);
  $('#result-range').textContent=`${money(result.p10)} — ${money(result.p90)}`;
  $('#result-liability').textContent=`${(result.liabilityRate*100).toFixed(1)}%`;
  $('#result-zero-rate').textContent=`${(result.zeroRate*100).toFixed(1)}%`;
  $('#result-limit-rate').textContent=`${(result.policyLimitRate*100).toFixed(1)}%`;
  $('#result-count').textContent=number.format(result.iterations);
  $('#result-updated').textContent=complete?`Completed ${new Date().toLocaleTimeString([], {hour:'numeric',minute:'2-digit'})}`:'Live · updating from fresh draws';
  $('#result-p10').textContent=money(result.p10);$('#result-p50').textContent=money(result.p50);$('#result-p90').textContent=money(result.p90);$('#result-mean').textContent=money(result.mean);
  renderChart(result.outcomes);
  renderFactorGraph(result);
  const focusList=$('#investigation-focus-list');
  if(focusList) focusList.replaceChildren(...result.focus.slice(0,5).map((item,index)=>{
    const row=document.createElement('article');row.className='focus-row';
    const rank=document.createElement('span');rank.className='focus-rank';rank.textContent=String(index+1).padStart(2,'0');
    const body=document.createElement('div');body.className='focus-body';
    const title=document.createElement('strong');title.textContent=item.label;
    const score=document.createElement('span');score.className='focus-score';score.textContent=`Relative priority ${item.priorityScore}/100`;
    const swing=document.createElement('p');swing.textContent=`Conditional mean: ${money(item.meanWithFactor)} with factor vs ${money(item.meanWithoutFactor)} without · ${money(item.conditionalSwing)} swing · factor present in ${Math.round(item.probability*100)}% of trials`;
    const action=document.createElement('p');action.className='focus-action';action.textContent=item.action;
    body.append(title,score,swing,action);row.append(rank,body);
    const target=(window.caseRecords||[]).find(record=>{
      const haystack=`${record.title} ${record.description}`.toLowerCase();
      const needles={liability:['three different accounts','scene photographs','police report'],scope:['vehicle assignment','dispatch'],medical:['itemized','medical ledger','specials'],wages:['commission records','1099','wage'],surgery:['surgery','orthopaedic'],comparativeFault:['liability note','collision','incident accounts']};
      return (needles[item.key]||[]).some(needle=>haystack.includes(needle));
    });
    if(target){const open=document.createElement('button');open.type='button';open.textContent='Open related record ↗';open.addEventListener('click',()=>window.dispatchEvent(new CustomEvent('case:open-record',{detail:target})));body.append(open);}
    return row;
  }));
  const posteriorLine=$('#posterior-note');
  if(posteriorLine) posteriorLine.textContent=`Evidence-adjusted rates · mechanism ${Math.round(result.pLiability*100)}% · scope ${Math.round(result.pScope*100)}% · specials ${Math.round(result.pSpecials*100)}% · wages ${Math.round(result.pWages*100)}% · future surgery ${Math.round(result.pSurgery*100)}%`;
}
async function run() {
  clearTimeout(runTimer);
  cancelSimulation?.();
  const runId=++activeRun;
  const button=$('#run-simulation');
  button.disabled=true;
  button.setAttribute('aria-busy','true');
  button.textContent=`Simulating 0 / ${number.format(SIMULATION_ITERATIONS)}…`;
  $('#result-updated').textContent='Starting one million trials…';
  try {
    return await new Promise((resolve,reject)=>{
      const worker=new Worker(new URL('./simulation-worker.mjs?build=20261002-million',import.meta.url),{type:'module'});
      const finish=(result,error)=>{
        worker.terminate();
        if(runId===activeRun)cancelSimulation=undefined;
        if(error)reject(error);else resolve(result);
      };
      cancelSimulation=()=>finish(undefined);
      worker.onmessage=({data})=>{
        if(runId!==activeRun)return;
        if(data.type==='error'){finish(undefined,new Error(data.message));return;}
        if(data.type==='progress'){
          button.textContent=`Simulating ${number.format(data.completed)} / ${number.format(data.total)}…`;
          $('#result-updated').textContent=`${number.format(data.completed)} of ${number.format(data.total)} trials · live`;
          return;
        }
        if(data.type==='result'){
          renderResult(data.result,data.complete);
          if(data.complete)finish(data.result);
        }
      };
      worker.onerror=event=>finish(undefined,new Error(event.message||'Simulation worker failed.'));
      worker.postMessage({assumptions:{...state.assumptions},facts:state.facts.map(fact=>({...fact}))});
    });
  } catch(error){
    if(runId===activeRun)$('#result-updated').textContent=`Simulation failed: ${error.message}`;
    throw error;
  } finally {
    if(runId===activeRun){
      button.disabled=false;
      button.removeAttribute('aria-busy');
      button.innerHTML='Run 1,000,000 simulations <span>↗</span>';
    }
  }
}
let runTimer;
function scheduleRun(){clearTimeout(runTimer);cancelSimulation?.();activeRun++;runTimer=setTimeout(run,180);}

function renderEvidence(records) {
  const query=$('#record-search').value.trim().toLowerCase();
  const kind=$('#record-type-filter').value;
  const analyst=state.facts.map((fact,index)=>({id:`investigator-${fact.id}`,key:'investigator',type:'Investigator fact',title:fact.text,description:`${dimensions[fact.dimension]??fact.dimension} · LR ${Number(fact.lr).toFixed(2)} · ${Math.round(fact.confidence*100)}% confidence`,date:new Date(Date.UTC(2026,9,2+index)),dateLabel:'02 Oct 2026',source:fact.source||'Analyst supplied',path:null,raw:fact}));
  const all=[...records,...analyst];
  const filtered=all.filter(record=>(kind==='all'||record.key===kind)&&(!query||`${record.title} ${record.description} ${record.source}`.toLowerCase().includes(query))).slice(0,250);
  $('#evidence-count').textContent=number.format(all.length);
  $('#evidence-result-count').textContent=`${number.format(filtered.length)} of ${number.format(all.length)} records`;
  const tbody=$('#evidence-table-body');
  tbody.replaceChildren(...filtered.map(record=>{
    const tr=document.createElement('tr');
    const name=document.createElement('td');name.textContent=record.title;
    if(record.description){const detail=document.createElement('small');detail.className='record-description';detail.textContent=record.description;name.append(detail);}
    const type=document.createElement('td');type.className='record-type';type.textContent=record.type;
    const date=document.createElement('td');date.textContent=record.dateLabel;
    const source=document.createElement('td');source.textContent=record.source;
    const review=document.createElement('td');const badge=document.createElement('span');badge.className=`review-state${record.key==='investigator'?' open':''}`;badge.textContent=record.key==='investigator'?'Analyst input':/task|expenses/i.test(record.type)?'Follow-up':'Source record';review.append(badge);
    const action=document.createElement('td');
    if(record.key==='investigator'){
      const btn=document.createElement('button');btn.className='record-open';btn.type='button';btn.textContent='Edit ↗';btn.addEventListener('click',()=>openFactEditor(record.raw));action.append(btn);
    }else{
      const btn=document.createElement('button');btn.className='record-open';btn.type='button';btn.textContent='Open ↗';btn.addEventListener('click',()=>window.dispatchEvent(new CustomEvent('case:open-record',{detail:record})));action.append(btn);
    }
    tr.append(name,type,date,source,review,action);return tr;
  }));
}

export function initializeInvestigatorWorkspace({records, rawData}) {
  window.caseRecords=records;
  const tabs=[...document.querySelectorAll('.nav-tab')];
  const pages=[...document.querySelectorAll('[data-panel]')];
  function activate(name,updateHash=true){
    for(const tab of tabs){const active=tab.dataset.page===name;tab.setAttribute('aria-selected',String(active));tab.tabIndex=active?0:-1;}
    for(const page of pages){page.hidden=page.dataset.panel!==name;page.classList.toggle('active',page.dataset.panel===name);}
    const pageLabel=document.querySelector('#current-page-label');
    if(pageLabel)pageLabel.textContent=name==='assistant'?'LOCAL AGENT':name.replaceAll('-',' ').toUpperCase();
    if(updateHash) history.replaceState(null,'',`#${name}`);
    window.dispatchEvent(new CustomEvent('workspace:page-change',{detail:{page:name}}));
  }
  navigateWorkspace=activate;
  const factDimensions=Object.keys(dimensions);
  function normalizeFact(input, current={}) {
    const text=String(input.text??current.text??'').trim();
    const source=String(input.source??current.source??'').trim();
    const dimension=String(input.dimension??current.dimension??'liability');
    const lr=Number(input.lr??current.lr??1),confidence=Number(input.confidence??current.confidence??.5);
    if(!text||text.length>180)throw new Error('Fact text must contain 1–180 characters.');
    if(source.length>100)throw new Error('Source reference must be 100 characters or fewer.');
    if(!factDimensions.includes(dimension))throw new Error('Choose a supported analyst fact dimension.');
    if(!Number.isFinite(lr)||lr<.25||lr>4)throw new Error('Likelihood ratio must be between 0.25 and 4.');
    if(!Number.isFinite(confidence)||confidence<0||confidence>1)throw new Error('Confidence must be between 0 and 1.');
    return {text,source,dimension,lr,confidence};
  }
  function refreshAfterInputChange(){
    save();renderAssumptions();renderFacts();renderEvidence(records);renderAnomalies();
  }
  function modelSnapshot(){
    return {assumptions:{...state.assumptions},facts:state.facts.map(({id,text,source,dimension,lr,confidence})=>({id,text,source,dimension,lr,confidence})),
      latestResult:lastResult?{iterations:lastResult.iterations,mean:lastResult.mean,p10:lastResult.p10,p50:lastResult.p50,p90:lastResult.p90,conditionalP50:lastResult.conditionalP50,liabilityRate:lastResult.liabilityRate,zeroRate:lastResult.zeroRate,policyLimitRate:lastResult.policyLimitRate,gateDependence:lastResult.gateDependence,focus:lastResult.focus,pLiability:lastResult.pLiability,pScope:lastResult.pScope,pSpecials:lastResult.pSpecials,pWages:lastResult.pWages,pSurgery:lastResult.pSurgery}:null};
  }
  window.caseAgentTools={
    getOverview(){
      const matter=rawData.matter?.body??{};
      return {matter:{id:matter.id??'00001-SAPINI',name:matter.display_number??matter.name??'Justin Sapini',status:matter.status??'Unknown',stage:matter.matter_stage?.id??rawData.matter_stages?.sapini_stage??'Unknown'},recordCount:records.length,recordTypes:[...new Set(records.map(record=>record.type))],model:modelSnapshot()};
    },
    searchRecords({query='',type='any',limit=10}={}){
      const needle=String(query).trim().toLowerCase();
      const allowed=new Set(['any','documents','notes','communications','tasks','calendar_entries','expenses']);
      if(!allowed.has(type))throw new Error('Unsupported case record type.');
      const safeLimit=Math.max(1,Math.min(20,Math.floor(Number(limit)||10)));
      const terms=needle.match(/[\p{L}\p{N}]{3,}/gu)??[];
      return records.map(record=>{
        const haystack=`${record.title} ${record.description} ${record.type}`.toLowerCase();
        const matches=terms.filter(term=>haystack.includes(term)).length;
        const phrase=needle&&haystack.includes(needle)?1:0;
        return {record,matches,phrase};
      }).filter(({record,matches})=>(type==='any'||record.key===type)&&(!terms.length||matches>0))
        .sort((a,b)=>b.phrase-a.phrase||b.matches-a.matches||b.record.date-a.record.date).slice(0,safeLimit)
        .map(({record:{id,key,type,title,description,dateLabel,source}})=>({id,type:key,displayType:type,title,description:description.slice(0,1600),date:dateLabel,source}));
    },
    openRecord({record_id}){
      const record=records.find(item=>item.id===String(record_id));
      if(!record)throw new Error('Record not found. Search records first and use a returned record ID.');
      window.dispatchEvent(new CustomEvent('case:open-record',{detail:record}));
      return {opened:true,record:{id:record.id,type:record.type,title:record.title,date:record.dateLabel,source:record.source}};
    },
    getModel(){return modelSnapshot();},
    getOutcomes(){return modelSnapshot().latestResult;},
    async addFact(input){
      const values=normalizeFact(input);
      const fact={id:crypto.randomUUID(),...values,seeded:false};
      state.facts.unshift(fact);refreshAfterInputChange();
      const result=await run();
      return {saved:true,fact:{id:fact.id,text:fact.text,source:fact.source,dimension:fact.dimension,lr:fact.lr,confidence:fact.confidence},simulationCompleted:!!result,persistence:'Browser local storage'};
    },
    async updateFact(input){
      const fact=state.facts.find(item=>item.id===String(input.id));
      if(!fact)throw new Error('Analyst fact not found. Use get_case_model to see fact IDs.');
      Object.assign(fact,normalizeFact(input,fact));refreshAfterInputChange();
      const result=await run();
      return {updated:true,fact:{id:fact.id,text:fact.text,source:fact.source,dimension:fact.dimension,lr:fact.lr,confidence:fact.confidence},simulationCompleted:!!result,persistence:'Browser local storage'};
    },
    async updateAssumption({key,value}){
      const control=controls.find(item=>item.key===String(key));
      if(!control)throw new Error('Unsupported assumption key. Use get_case_model to see editable assumptions.');
      const numeric=Number(value);
      if(!Number.isFinite(numeric)||numeric<control.min||numeric>control.max)throw new Error(`${control.label} must be between ${control.min} and ${control.max}.`);
      if(key==='futureLow'&&numeric>state.assumptions.futureHigh)throw new Error('Future care low cannot exceed future care high.');
      if(key==='futureHigh'&&numeric<state.assumptions.futureLow)throw new Error('Future care high cannot be below future care low.');
      state.assumptions[key]=numeric;save();renderAssumptions();
      const result=await run();
      return {updated:true,key,value:numeric,simulationCompleted:!!result,persistence:'Browser local storage'};
    },
    async runSimulation(){
      const result=await run();
      return {completed:!!result,...modelSnapshot()};
    },
  };
  tabs.forEach((tab,index)=>{
    tab.addEventListener('click',()=>activate(tab.dataset.page));
    tab.addEventListener('keydown',event=>{
      if(!['ArrowDown','ArrowUp','ArrowRight','ArrowLeft','Home','End'].includes(event.key)) return;
      event.preventDefault();
      const next=event.key==='Home'?0:event.key==='End'?tabs.length-1:(index+(event.key==='ArrowDown'||event.key==='ArrowRight'?1:-1)+tabs.length)%tabs.length;
      tabs[next].focus();tabs[next].click();
    });
  });
  document.querySelectorAll('[data-go]').forEach(button=>button.addEventListener('click',()=>{
    activate(button.dataset.go);
    if(button.dataset.go==='facts') requestAnimationFrame(()=>$('#fact-form input[name="text"]').focus());
  }));
  activate(location.hash.slice(1)||'overview',false);

  $('#record-search').addEventListener('input',()=>renderEvidence(records));
  $('#record-type-filter').addEventListener('change',()=>renderEvidence(records));
  $('#fact-confidence').addEventListener('input',event=>$('#fact-confidence-label').textContent=`${Math.round(event.target.value*100)}%`);
  $('#fact-form').addEventListener('submit',event=>{
    event.preventDefault();
    const form=new FormData(event.currentTarget);
    const text=String(form.get('text')||'').trim();if(!text)return;
    const values={text,source:String(form.get('source')||'').trim(),dimension:String(form.get('dimension')),lr:clamp(form.get('lr')||1,.25,4),confidence:clamp(form.get('confidence')||.5,0,1)};
    if(editingFactId){Object.assign(state.facts.find(item=>item.id===editingFactId)||{},values);}
    else state.facts.unshift({id:crypto.randomUUID(),...values,seeded:false});
    editingFactId=null;save();renderFacts();renderEvidence(records);renderAnomalies();run();event.currentTarget.reset();$('#fact-confidence-label').textContent='50%';$('#fact-form-title').textContent='Add a fact';event.currentTarget.querySelector('[type="submit"]').textContent='Save investigator fact';$('#fact-cancel').hidden=true;
  });
  $('#fact-cancel').addEventListener('click',()=>{editingFactId=null;$('#fact-form').reset();$('#fact-confidence-label').textContent='50%';$('#fact-form-title').textContent='Add a fact';$('#fact-form').querySelector('[type="submit"]').textContent='Save investigator fact';$('#fact-cancel').hidden=true;});
  $('#reset-model').addEventListener('click',()=>{Object.assign(state.assumptions,defaults.assumptions);state.facts=structuredClone(defaults.facts);save();renderAssumptions();renderFacts();renderEvidence(records);renderAnomalies();run();});
  $('#run-simulation').addEventListener('click',run);
  renderAssumptions();renderFacts();renderAnomalies();renderEvidence(records);run();
}
