import { initializeCaseChat } from './chat.mjs?build=20261002-million';
import { initializeInvestigatorWorkspace } from './analysis-model.mjs?build=20261002-million';

const categoryConfig = [
  { key: 'notes', label: 'Notes', singular: 'Note', color: '#181818', radius: 3.45 },
  { key: 'communications', label: 'Communications', singular: 'Communication', color: '#3d3d3d', radius: 4.9 },
  { key: 'tasks', label: 'Tasks', singular: 'Task', color: '#626262', radius: 6.35 },
  { key: 'calendar_entries', label: 'Calendar', singular: 'Calendar event', color: '#858585', radius: 7.8 },
  { key: 'documents', label: 'Documents', singular: 'Document', color: '#a9a9a9', radius: 9.25 },
  { key: 'expenses', label: 'Expenses', singular: 'Expense', color: '#c7c7c7', radius: 10.7 },
];

const $ = (selector) => document.querySelector(selector);
const numberFormat = new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 });
const dateFormat = new Intl.DateTimeFormat('en-US', { day: '2-digit', month: 'short', year: 'numeric', timeZone: 'UTC' });
let rawData;
let records = [];
let cursorIndex = 0;
let playing = false;
let playTimer;
let selectedRecord;
let renderer;
let controls;
let scene;
let camera;
let orbitLayers = [];
let nodeMeshes = [];
let focusLine;
let raycaster;
let pointer;
let activeCategories = new Set(categoryConfig.map((item) => item.key));
let hoveredMesh;

function fieldValues(data) {
  const names = new Map((data.custom_fields?.items ?? []).map((field) => [field.body.name, `{{field:${field.body.name}}}`]));
  const output = {};
  for (const item of data.matter?.body?.custom_field_values ?? []) {
    const name = [...names.entries()].find(([, token]) => token === item.custom_field?.id)?.[0];
    if (name) output[name] = item.value;
  }
  return output;
}

function cleanDate(value) {
  if (!value) return new Date('2023-05-07T12:00:00');
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? new Date('2023-05-07T12:00:00') : date;
}

function flattenRecords(data) {
  const flattened = [];
  for (const category of categoryConfig) {
    for (const [index, item] of (data[category.key]?.items ?? []).entries()) {
      const body = item.body ?? item;
      const dateValue = body.date ?? body.due_at ?? body.start_at ?? body.received_at;
      const title = body.subject ?? body.summary ?? body.name ?? body.note?.split('\n')[0] ?? `${category.label} ${index + 1}`;
      const description = body.detail ?? body.body ?? body.description ?? body.note ?? item.description ?? '';
      const date = cleanDate(dateValue);
      flattened.push({
        id: `${category.key}-${index}`,
        key: category.key,
        type: category.singular,
        color: category.color,
        title: String(title).replace(/\s+/g, ' ').trim(),
        description: String(description).trim(),
        date,
        dateLabel: dateFormat.format(date),
        source: item.local_path ? 'Case document' : category.singular,
        path: item.local_path,
        raw: item,
        index,
      });
    }
  }
  return flattened.sort((a, b) => a.date - b.date || a.type.localeCompare(b.type) || a.index - b.index);
}

function getFieldValues() {
  const values = fieldValues(rawData);
  const matter = rawData.matter?.body ?? {};
  const stage = (matter.matter_stage?.id ?? rawData.matter_stages?.sapini_stage ?? 'Litigation').toString().replace(/[{}]/g, '').split(':').pop();
  $('#stage-label').textContent = stage.toUpperCase();
  const specials = Number(values['Medical Specials To Date'] ?? 0);
  $('#metric-specials').textContent = `$${numberFormat.format(specials)}`;
  const wageText = String(values['Wage Loss Claimed'] ?? '$214,000');
  const wageAmount = wageText.match(/\$\s?([\d,]+)/)?.[1] ?? '214,000';
  $('#metric-wages').textContent = `$${wageAmount}`;
  $('#metric-value').textContent = `$${numberFormat.format(Number(values['Estimated Case Value'] ?? 375000))}`;
  $('#metric-records').textContent = numberFormat.format(records.length);

  const summary = String(values['Case Summary'] ?? 'Personal injury matter in active litigation.');
  $('#case-summary').textContent = summary;
  const liability = String(values['Liability Assessment'] ?? 'Contested');
  $('#liability-value').textContent = liability.toLowerCase().includes('contested') ? 'Contested' : liability.slice(0, 40);
  $('#coverage-value').textContent = 'Confirmed · $100k / $300k';
  const lien = String(values['Health Insurance or Lien Holder'] ?? 'Medicaid lien, $22,180');
  $('#lien-value').textContent = lien.match(/\$\s?[\d,]+(?:\.\d{2})?/)?.[0] ?? '$22,180';
  const treatment = String(values['Treatment Status'] ?? 'Ongoing');
  $('#treatment-value').textContent = /active|ongoing/i.test(treatment) ? 'Ongoing' : treatment.slice(0, 30);
}

function setSelected(record) {
  if (!record) return;
  selectedRecord = record;
  $('#selected-type').textContent = record.type.toUpperCase().replace(/_/g, ' ');
  $('#selected-type').style.color = record.color;
  $('#selected-type').style.borderColor = `${record.color}55`;
  $('#selected-title').textContent = record.title;
  $('#selected-date').textContent = record.dateLabel;
  $('#selected-source').textContent = record.source;
  $('#selected-body').textContent = record.description || 'No additional detail is attached to this record.';
  const link = $('#source-link');
  if (record.path) {
    link.href = encodeURI(record.path);
    link.hidden = false;
  } else {
    link.hidden = true;
    link.removeAttribute('href');
  }
  updateFocusLine(record);
  for (const mesh of nodeMeshes) {
    const isSelected = mesh.userData.record.id === record.id;
    mesh.material.emissiveIntensity = isSelected ? 1.5 : 0.3;
    mesh.scale.setScalar(isSelected ? 1.75 : mesh.userData.baseScale);
  }
}

function findRecord(match, key = 'notes') {
  const search = String(match).toLowerCase();
  return records.find((record) => record.key === key && `${record.title} ${record.description}`.toLowerCase().includes(search));
}

function openRecord(record) {
  if (!record) return;
  if ($('#page-timeline').hidden) {
    $('#tab-timeline').click();
    requestAnimationFrame(() => openRecord(record));
    return;
  }
  cursorIndex = records.indexOf(record);
  $('#timeline').value = String(cursorIndex);
  updateTimeline();
  setSelected(record);
  if (controls) controls.target.set(0, 0, 0);
  $('#map-title').scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function appendText(parent, tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  element.textContent = text;
  parent.append(element);
  return element;
}

function renderTasks() {
  const tasks = (rawData.tasks?.items ?? []).map((item) => item.body ?? item)
    .filter((task) => !/complete|completed/i.test(String(task.status ?? '')))
    .sort((a, b) => cleanDate(a.due_at).getTime() - cleanDate(b.due_at).getTime());
  const list = $('#task-list');
  $('#task-count').textContent = String(tasks.length);
  const today = new Date();
  today.setUTCHours(0, 0, 0, 0);
  list.replaceChildren(...tasks.map((task) => {
    const row = document.createElement('article');
    row.className = 'task-row';
    const due = cleanDate(task.due_at);
    const dayDelta = Math.floor((Date.UTC(due.getUTCFullYear(), due.getUTCMonth(), due.getUTCDate()) - today.getTime()) / 86400000);
    const date = document.createElement('div');
    date.className = `task-date${dayDelta < 0 ? ' overdue' : ''}`;
    appendText(date, 'span', '', dateFormat.format(due));
    appendText(date, 'small', '', 'Due date');
    row.append(date);
    const main = document.createElement('div');
    main.className = 'task-main';
    appendText(main, 'strong', '', task.name ?? 'Untitled task');
    if (task.description) appendText(main, 'p', '', task.description);
    const taskRecord = findRecord(task.name ?? '', 'tasks');
    if (taskRecord) {
      const open = appendText(main, 'button', 'task-open', 'OPEN TASK RECORD ↗');
      open.type = 'button';
      open.addEventListener('click', () => openRecord(taskRecord));
    }
    row.append(main);
    const status = document.createElement('span');
    status.className = `task-status ${dayDelta < 0 ? 'overdue' : dayDelta <= 7 ? 'soon' : 'upcoming'}`;
    status.textContent = dayDelta < 0 ? `${Math.abs(dayDelta)} days overdue` : dayDelta === 0 ? 'Due today' : dayDelta <= 7 ? 'Due soon' : 'Open';
    row.append(status);
    row.dataset.taskName = task.name ?? '';
    return row;
  }));
}

function renderEvents() {
  const startOfToday = new Date();
  startOfToday.setUTCHours(0, 0, 0, 0);
  const events = (rawData.calendar_entries?.items ?? []).map((item) => item.body ?? item)
    .filter((event) => event.start_at && cleanDate(event.start_at) >= startOfToday)
    .sort((a, b) => cleanDate(a.start_at) - cleanDate(b.start_at))
    .slice(0, 5);
  const list = $('#event-list');
  list.replaceChildren(...events.map((event) => {
    const row = document.createElement('article');
    row.className = 'event-row';
    const date = document.createElement('div');
    date.className = 'event-date';
    const dateValue = cleanDate(event.start_at);
    appendText(date, 'span', '', dateFormat.format(dateValue));
    row.append(date);
    const body = document.createElement('div');
    appendText(body, 'strong', '', event.summary ?? 'Calendar event');
    if (event.description) appendText(body, 'p', '', event.description);
    row.append(body);
    return row;
  }));
  if (!events.length) appendText(list, 'p', 'empty-state', 'No upcoming events in the snapshot.');
}

function renderIssues() {
  const issues = [
    { title: 'Liability account needs reconciliation', detail: 'The file contains differing collision accounts; no police accident report is recorded. Scene photos were offered but not requested.', needle: 'three different accounts', key: 'notes' },
    { title: 'Scope of employment remains unresolved', detail: 'Vehicle assignment and dispatch records are still outstanding; the file identifies them as central to Metro-North liability.', needle: 'vehicle assignment and dispatch', key: 'notes' },
    { title: 'Wage-loss support is limited', detail: 'The $214,000 claim is based on a 2022 1099. Updated commission records are an overdue task.', needle: 'commission records', key: 'tasks' },
    { title: 'Treatment charges remain open', detail: 'Updated chiropractic and physical therapy notes and itemized ledgers are still requested.', needle: 'reconcile chiropractic and pt ledgers', key: 'tasks' },
    { title: 'Prior injury history needs reconciliation', detail: 'An intake response and older ankle imaging differ; clarify the history before using it in analysis.', needle: 'prior injury discrepancy', key: 'notes' },
  ];
  if ($('#issue-count')) $('#issue-count').textContent = String(issues.length);
  const lists = [$('#issue-list'), $('#evidence-issues')].filter(Boolean);
  for (const list of lists) list.replaceChildren(...issues.map((issue) => {
    const row = document.createElement('article');
    row.className = 'issue';
    const dot = document.createElement('span');
    dot.className = 'issue-dot';
    row.append(dot);
    const body = document.createElement('div');
    appendText(body, 'strong', '', issue.title);
    appendText(body, 'p', '', issue.detail);
    const source = findRecord(issue.needle, issue.key);
    if (source) {
      const button = appendText(body, 'button', 'issue-link', `Open ${source.type.toLowerCase()} · ${source.dateLabel} ↗`);
      button.type = 'button';
      button.addEventListener('click', () => openRecord(source));
    }
    row.append(body);
    return row;
  }));
}

function setPriorityReview() {
  const taskRecord = findRecord('right shoulder surgical date', 'tasks');
  const note = findRecord('right shoulder surgery still has no date');
  const dateLabel = taskRecord?.dateLabel;
  $('#priority-title').textContent = 'Confirm the right shoulder surgery date';
  $('#priority-copy').textContent = `No confirmed date after three written requests. Follow-up task due ${dateLabel ?? 'in the matter task list'}${taskRecord && taskRecord.date < new Date() ? ' · overdue' : ''}.`;
  $('#priority-action').addEventListener('click', () => openRecord(note ?? taskRecord));
}

function populateCaseReview() {
  renderTasks();
  renderEvents();
  renderIssues();
  setPriorityReview();
}

function renderLegend() {
  const legend = $('#legend');
  legend.replaceChildren(...categoryConfig.map((category) => {
    const button = document.createElement('button');
    button.className = 'legend-button';
    button.type = 'button';
    button.setAttribute('aria-pressed', 'true');
    button.innerHTML = `<i class="legend-dot" style="color:${category.color};background:${category.color}"></i>${category.label}`;
    button.addEventListener('click', () => {
      if (activeCategories.has(category.key)) activeCategories.delete(category.key);
      else activeCategories.add(category.key);
      button.setAttribute('aria-pressed', String(activeCategories.has(category.key)));
      updateTimeline();
    });
    return button;
  }));
}

function buildScene(THREE, OrbitControls) {
  const host = $('#scene');
  const width = host.clientWidth;
  const height = host.clientHeight;
  if (!width || !height) return;
  if (!document.createElement('canvas').getContext('webgl2')) throw new Error('WebGL 2 is not available.');

  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(42, width / height, 0.1, 100);
  camera.position.set(0, 12.5, 23.5);
  camera.lookAt(0, 0, 0);
  renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.75));
  renderer.setSize(width, height);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  host.replaceChildren(renderer.domElement);

  scene.add(new THREE.AmbientLight(0xb6c2d1, 1.5));
  const light = new THREE.PointLight(0xffffff, 18, 34);
  light.position.set(-2, 5, 4);
  scene.add(light);

  const core = new THREE.Mesh(
    new THREE.IcosahedronGeometry(0.88, 2),
    new THREE.MeshStandardMaterial({ color: 0xe6e6e6, roughness: 0.7, metalness: 0.05 }),
  );
  scene.add(core);
  const coreWire = new THREE.Mesh(new THREE.IcosahedronGeometry(0.94, 1), new THREE.MeshBasicMaterial({ color: 0x555555, wireframe: true, transparent: true, opacity: 0.22 }));
  scene.add(coreWire);
  const coreRing = new THREE.Mesh(new THREE.TorusGeometry(1.35, 0.009, 7, 100), new THREE.MeshBasicMaterial({ color: 0x777777, transparent: true, opacity: 0.3 }));
  coreRing.rotation.x = Math.PI / 2.25;
  scene.add(coreRing);

  orbitLayers = [];
  nodeMeshes = [];
  const minTime = records[0].date.getTime();
  const maxTime = records.at(-1).date.getTime();
  for (const [laneIndex, category] of categoryConfig.entries()) {
    const group = new THREE.Group();
    group.position.y = (laneIndex - 2.5) * 0.22;
    scene.add(group);
    orbitLayers.push({ group, direction: 0, speed: 0 });

    const ring = new THREE.Mesh(
      new THREE.TorusGeometry(category.radius, 0.012, 6, 180),
      new THREE.MeshBasicMaterial({ color: category.color, transparent: true, opacity: 0.22 }),
    );
    ring.rotation.x = Math.PI / 2;
    group.add(ring);

    const ticks = new THREE.Group();
    for (let tick = 0; tick < 4; tick += 1) {
      const angle = -Math.PI / 2 + (tick / 3) * Math.PI * 2;
      const mark = new THREE.Mesh(new THREE.SphereGeometry(0.038, 8, 8), new THREE.MeshBasicMaterial({ color: 0x8794aa, transparent: true, opacity: 0.48 }));
      mark.position.set(Math.cos(angle) * category.radius, 0, Math.sin(angle) * category.radius);
      ticks.add(mark);
    }
    group.add(ticks);

    const categoryRecords = records.filter((record) => record.key === category.key);
    for (const record of categoryRecords) {
      const t = maxTime === minTime ? 0.5 : (record.date.getTime() - minTime) / (maxTime - minTime);
      const angle = -Math.PI / 2 + t * Math.PI * 2;
      const variation = ((record.index % 5) - 2) * 0.055;
      const radius = category.radius + variation;
      const size = category.key === 'tasks' || category.key === 'expenses' ? 0.12 : category.key === 'documents' ? 0.09 : 0.1;
      const material = new THREE.MeshStandardMaterial({ color: category.color, emissive: category.color, emissiveIntensity: 0.12, roughness: 0.55, metalness: 0.05, transparent: true, opacity: 0.96 });
      const mesh = new THREE.Mesh(new THREE.SphereGeometry(size, 12, 12), material);
      mesh.position.set(Math.cos(angle) * radius, Math.sin(angle * 2) * 0.1, Math.sin(angle) * radius);
      mesh.userData = { record, baseScale: 1 };
      group.add(mesh);
      nodeMeshes.push(mesh);
    }
  }

  const focusGeometry = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 0, 0)]);
  focusLine = new THREE.Line(focusGeometry, new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.58 }));
  scene.add(focusLine);

  raycaster = new THREE.Raycaster();
  raycaster.params.Points.threshold = 0.2;
  pointer = new THREE.Vector2();
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.06;
  controls.enablePan = false;
  controls.minDistance = 16;
  controls.maxDistance = 31;
  controls.minPolarAngle = 0.35;
  controls.maxPolarAngle = Math.PI / 2.05;
  controls.target.set(0, 0, 0);
  setSelected(selectedRecord);

  let downPoint;
  renderer.domElement.addEventListener('pointerdown', (event) => { downPoint = [event.clientX, event.clientY]; });
  renderer.domElement.addEventListener('pointerup', (event) => {
    if (!downPoint || Math.hypot(event.clientX - downPoint[0], event.clientY - downPoint[1]) > 6) return;
    const bounds = renderer.domElement.getBoundingClientRect();
    pointer.x = ((event.clientX - bounds.left) / bounds.width) * 2 - 1;
    pointer.y = -((event.clientY - bounds.top) / bounds.height) * 2 + 1;
    raycaster.setFromCamera(pointer, camera);
    const hit = raycaster.intersectObjects(nodeMeshes, false)[0];
    if (hit) setSelected(hit.object.userData.record);
    downPoint = undefined;
  });
  renderer.domElement.addEventListener('pointermove', (event) => {
    const bounds = renderer.domElement.getBoundingClientRect();
    pointer.x = ((event.clientX - bounds.left) / bounds.width) * 2 - 1;
    pointer.y = -((event.clientY - bounds.top) / bounds.height) * 2 + 1;
    raycaster.setFromCamera(pointer, camera);
    const hit = raycaster.intersectObjects(nodeMeshes.filter((mesh) => mesh.visible), false)[0];
    if (hit) {
      hoveredMesh = hit.object;
      renderer.domElement.style.cursor = 'pointer';
      const card = $('#hover-card');
      card.innerHTML = `<strong>${escapeHtml(hit.object.userData.record.title)}</strong>${hit.object.userData.record.dateLabel} · ${hit.object.userData.record.type}`;
      card.style.left = `${Math.min(event.offsetX + 13, bounds.width - 240)}px`;
      card.style.top = `${Math.max(event.offsetY - 12, 10)}px`;
      card.hidden = false;
    } else {
      hoveredMesh = undefined;
      renderer.domElement.style.cursor = 'grab';
      $('#hover-card').hidden = true;
    }
  });
  renderer.domElement.addEventListener('pointerleave', () => { $('#hover-card').hidden = true; renderer.domElement.style.cursor = 'grab'; });

  const resizeObserver = new ResizeObserver(() => {
    if (!renderer) return;
    const nextWidth = host.clientWidth;
    const nextHeight = host.clientHeight;
    if (!nextWidth || !nextHeight) return;
    camera.aspect = nextWidth / nextHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(nextWidth, nextHeight);
  });
  resizeObserver.observe(host);

  const clock = new THREE.Clock();
  const animate = () => {
    requestAnimationFrame(animate);
    const elapsed = clock.getDelta();
    for (const layer of orbitLayers) layer.group.rotation.y += elapsed * layer.speed * layer.direction;
    controls.update();
    renderer.render(scene, camera);
  };
  animate();
  updateTimeline();
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (character) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[character]);
}

function updateFocusLine(record) {
  if (!focusLine || !record) return;
  const mesh = nodeMeshes.find((node) => node.userData.record.id === record.id);
  if (!mesh) return;
  const worldPosition = new (mesh.position.constructor)();
  mesh.getWorldPosition(worldPosition);
  const positions = new Float32Array([0, 0, 0, worldPosition.x, worldPosition.y, worldPosition.z]);
  focusLine.geometry.setAttribute('position', new (focusLine.geometry.attributes.position.constructor)(positions, 3));
  focusLine.geometry.attributes.position.needsUpdate = true;
  focusLine.material.color.set(record.color);
}

function updateTimeline() {
  if (!records.length) return;
  cursorIndex = Math.min(Math.max(cursorIndex, 0), records.length - 1);
  const slider = $('#timeline');
  slider.max = String(records.length - 1);
  slider.value = String(cursorIndex);
  const date = records[cursorIndex].date;
  $('#scrub-date').textContent = `THROUGH ${dateFormat.format(date).toUpperCase()}`;
  let visible = 0;
  for (const mesh of nodeMeshes) {
    const record = mesh.userData.record;
    const show = activeCategories.has(record.key) && record.date <= date;
    mesh.visible = show;
    if (show) visible += 1;
  }
  $('#visible-count').textContent = numberFormat.format(visible);
  if (selectedRecord && selectedRecord.date > date) {
    const lastVisible = [...records].reverse().find((record) => record.date <= date && activeCategories.has(record.key));
    if (lastVisible) setSelected(lastVisible);
  } else if (!selectedRecord) {
    const initial = records.find((record) => record.key === 'notes' && /right shoulder surgery still has no date/i.test(record.title)) ?? records[cursorIndex];
    setSelected(initial);
  }
  updateFocusLine(selectedRecord);
}

function addProviderPreview() {
  const modal = $('#provider-modal');
  $('#provider-toggle').addEventListener('click', () => { modal.hidden = false; $('#provider-close').focus(); });
  $('#provider-close').addEventListener('click', () => { modal.hidden = true; $('#provider-toggle').focus(); });
  modal.addEventListener('click', (event) => { if (event.target === modal) modal.hidden = true; });
  document.addEventListener('keydown', (event) => { if (event.key === 'Escape' && !modal.hidden) modal.hidden = true; });
}

let sceneStarted = false;
async function ensureTimelineScene() {
  if (sceneStarted || !rawData || !$('#scene').clientWidth) return;
  sceneStarted = true;
  try {
    const THREE = await import('three');
    const { OrbitControls } = await import('three/addons/controls/OrbitControls.js');
    buildScene(THREE, OrbitControls);
  } catch (error) {
    console.warn('3D view unavailable:', error);
    $('#fallback').hidden = false;
    const current = records[Math.max(0, cursorIndex)];
    if (current) $('#visible-count').textContent = numberFormat.format(records.filter((record) => record.date <= current.date).length);
    $('#scrub-date').textContent = current ? `THROUGH ${dateFormat.format(current.date).toUpperCase()}` : 'Timeline unavailable';
  }
}

function setUpPlayback() {
  const button = $('#playback');
  button.addEventListener('click', () => {
    if (playing) {
      clearInterval(playTimer);
      playing = false;
      button.classList.remove('playing');
      button.textContent = '▶ PLAY THE CASE';
      return;
    }
    if (cursorIndex >= records.length - 1) cursorIndex = 0;
    playing = true;
    button.classList.add('playing');
    button.textContent = 'Ⅱ PAUSE STORY';
    updateTimeline();
    playTimer = setInterval(() => {
      cursorIndex = Math.min(cursorIndex + 2, records.length - 1);
      updateTimeline();
      if (cursorIndex === records.length - 1) {
        clearInterval(playTimer);
        playing = false;
        button.classList.remove('playing');
        button.textContent = '↻ REPLAY THE CASE';
      }
    }, 130);
  });
  $('#jump-now').addEventListener('click', () => {
    const today = new Date();
    const index = records.findLastIndex((record) => record.date <= today);
    cursorIndex = Math.max(0, index);
    updateTimeline();
  });
  $('#timeline').addEventListener('input', (event) => {
    if (playing) {
      clearInterval(playTimer);
      playing = false;
      button.classList.remove('playing');
      button.textContent = '▶ PLAY THE CASE';
    }
    cursorIndex = Number(event.target.value);
    updateTimeline();
  });
}

async function main() {
  try {
    const response = await fetch('./sapini-clio-data.json');
    if (!response.ok) throw new Error(`Could not load case snapshot (${response.status}).`);
    rawData = await response.json();
    records = flattenRecords(rawData);
    getFieldValues();
    $('#metric-records-timeline').textContent = numberFormat.format(records.length);
    renderLegend();
    populateCaseReview();
    initializeCaseChat(rawData).catch((error) => {
      console.error('Case chat initialization failed:', error);
      document.querySelector('#chat-status').textContent = 'Chat could not be initialized';
    });
    setUpPlayback();
    addProviderPreview();
    $('#timeline').max = String(Math.max(0, records.length - 1));
    const todayIndex = records.findLastIndex((record) => record.date <= new Date());
    cursorIndex = Math.max(0, todayIndex);
    const initial = records.find((record) => record.key === 'notes' && /right shoulder surgery still has no date/i.test(record.title)) ?? records[cursorIndex];
    setSelected(initial);
    initializeInvestigatorWorkspace({records, rawData});
    window.addEventListener('workspace:page-change', event => { if (event.detail.page === 'timeline') ensureTimelineScene(); });
    window.addEventListener('case:open-record', event => {
      document.querySelector('#tab-timeline').click();
      requestAnimationFrame(() => openRecord(event.detail));
    });
    if (document.querySelector('#tab-timeline').getAttribute('aria-selected') === 'true') ensureTimelineScene();
  } catch (error) {
    console.error(error);
    const startupError = document.querySelector('#startup-error');
    if (startupError) { startupError.hidden = false; startupError.textContent = `Case workspace could not finish loading: ${error?.message ?? error}`; }
    $('#selected-title').textContent = 'Could not load the case snapshot';
    $('#selected-body').textContent = 'Serve this folder over HTTP and make sure sapini-clio-data.json is beside index.html.';
    $('#fallback').hidden = false;
    $('#fallback').textContent = 'Use a local web server to load the case snapshot.';
  }
}

main();
