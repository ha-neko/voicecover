const form = document.querySelector('#dub-form');
const fileInput = document.querySelector('#video');
const fileLabel = document.querySelector('#file-label');
const dropzone = document.querySelector('#dropzone');
const urlField = document.querySelector('#url-field');
const youtubeInput = document.querySelector('#youtube-url');
const submit = document.querySelector('#submit');
const errorBox = document.querySelector('#form-error');
const processCard = document.querySelector('#process-card');
const caption = document.querySelector('#process-caption');
const consoleBox = document.querySelector('#console');
const result = document.querySelector('#result');
const targetInput = document.querySelector('[name=target_language]');
const voicePicker = document.querySelector('#voice-picker');
const voiceHelp = document.querySelector('#voice-help');
const profilePicker = document.querySelector('#profile-picker');
const profileHelp = document.querySelector('#profile-help');
const retryButton = document.querySelector('#retry');
const pauseButton = document.querySelector('#pause');
const discardButton = document.querySelector('#discard');
const progressTrack = document.querySelector('#progress-track');
const progressFill = document.querySelector('#progress-fill');
const progressPercent = document.querySelector('#progress-percent');
const progressLabel = document.querySelector('#progress-label');
const elapsedLabel = document.querySelector('#elapsed-label');
const elapsedTime = document.querySelector('#elapsed-time');
const startedTime = document.querySelector('#started-time');
const attemptCount = document.querySelector('#attempt-count');
let sourceMode = 'upload';
let pollTimer;
let voiceTimer;
let currentJob;
let profiles = [];

function formatDuration(totalSeconds) {
  const seconds = Math.max(0, Math.floor(totalSeconds));
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const remainder = seconds % 60;
  return hours > 0
    ? `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(remainder).padStart(2, '0')}`
    : `${String(minutes).padStart(2, '0')}:${String(remainder).padStart(2, '0')}`;
}

function renderTiming(job) {
  const started = job?.first_started_at ? new Date(job.first_started_at) : null;
  const validStart = started && !Number.isNaN(started.getTime());
  const activeStarted = job?.active_started_at ? new Date(job.active_started_at) : null;
  const validActiveStart = activeStarted && !Number.isNaN(activeStarted.getTime());
  const accumulated = Number.isFinite(Number(job?.active_seconds)) ? Number(job.active_seconds) : 0;
  const currentAttempt = job?.status === 'running' && validActiveStart
    ? Math.max(0, (new Date() - activeStarted) / 1000)
    : 0;
  elapsedLabel.textContent = job?.status === 'completed' ? 'Total active time' : 'Active time';
  elapsedTime.textContent = formatDuration(accumulated + currentAttempt);
  startedTime.textContent = validStart ? started.toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'}) : '—';
  attemptCount.textContent = job?.attempts ? String(job.attempts) : '—';
}

setInterval(() => { if (currentJob) renderTiming(currentJob); }, 1000);

document.querySelectorAll('.tab').forEach(tab => tab.addEventListener('click', () => {
  document.querySelectorAll('.tab').forEach(item => item.classList.toggle('active', item === tab));
  sourceMode = tab.dataset.source;
  dropzone.classList.toggle('hidden', sourceMode !== 'upload');
  urlField.classList.toggle('hidden', sourceMode !== 'youtube');
  if (sourceMode === 'upload') youtubeInput.value = ''; else fileInput.value = '';
}));

fileInput.addEventListener('change', () => {
  if (fileInput.files.length === 1) fileLabel.textContent = fileInput.files[0].name;
  if (fileInput.files.length > 1) fileLabel.textContent = `${fileInput.files.length} videos selected`;
});
['dragenter', 'dragover'].forEach(name => dropzone.addEventListener(name, event => { event.preventDefault(); dropzone.classList.add('drag'); }));
['dragleave', 'drop'].forEach(name => dropzone.addEventListener(name, event => { event.preventDefault(); dropzone.classList.remove('drag'); }));
dropzone.addEventListener('drop', event => {
  if (event.dataTransfer.files[0]) {
    fileInput.files = event.dataTransfer.files;
    fileLabel.textContent = event.dataTransfer.files.length === 1 ? event.dataTransfer.files[0].name : `${event.dataTransfer.files.length} videos selected`;
  }
});

const volume = document.querySelector('[name=original_volume]');
volume.addEventListener('input', () => document.querySelector('#volume-value').textContent = `${Math.round(volume.value * 100)}%`);

const cloneToggle = document.querySelector('[name=voice_clone]');
const modeInputs = [...document.querySelectorAll('[name=localization_mode]')];
const visualOnlyField = document.querySelector('[name=visual_only]');
const soundOnlyField = document.querySelector('[name=sound_only]');
const localizeVisualToggle = document.querySelector('[name=localize_visual_text]');
const visualContextToggle = document.querySelector('[name=visual_context]');
const writeVisualTrackToggle = document.querySelector('[name=write_visual_track]');
const cloneModelField = document.querySelector('#clone-model-field');
const submitCopy = submit.querySelector('span');

const stepCopy = {
  full: [
    ['Extract audio', 'Prepare the soundtrack'], ['Transcribe', 'Whisper speech recognition'],
    ['Translate', 'Speech and slide localization'], ['Generate voice', 'Synthesize translated speech'],
    ['Synchronize & mux', 'Build the finished video']
  ],
  sound: [
    ['Extract audio', 'Keep original video pixels'], ['Transcribe', 'Whisper speech recognition'],
    ['Translate speech', 'Localize every utterance'], ['Generate voice', 'Synthesize translated speech'],
    ['Synchronize & mux', 'Attach the new soundtrack']
  ],
  visual: [
    ['Sample frames', 'Keep the original soundtrack'], ['Read slides', 'OCR static on-screen text'],
    ['Translate text', 'Preserve Arabic and protected text'], ['Render overlays', 'Blend localized typography'],
    ['Export video', 'Copy the original audio']
  ]
};

function currentMode() {
  return modeInputs.find(input => input.checked)?.value || 'full';
}

function updateStepCopy(mode) {
  document.querySelectorAll('#steps li').forEach((step, index) => {
    const [title, detail] = stepCopy[mode][index];
    step.querySelector('strong').textContent = title;
    step.querySelector('small').textContent = detail;
  });
}

function setGroupEnabled(selector, enabled) {
  document.querySelectorAll(selector).forEach(group => {
    group.classList.toggle('mode-disabled', !enabled);
    group.setAttribute('aria-disabled', String(!enabled));
    group.querySelectorAll('input, select').forEach(control => { control.disabled = !enabled; });
  });
}

function syncCloneControls() {
  const speechEnabled = currentMode() !== 'visual';
  cloneModelField.classList.toggle('hidden', !cloneToggle.checked || !speechEnabled);
  voicePicker.disabled = !speechEnabled || cloneToggle.checked;
  if (!speechEnabled) {
    voiceHelp.textContent = 'Original soundtrack will be copied without speech synthesis.';
    return;
  }
  if (cloneToggle.checked) {
    voiceHelp.textContent = "F5 will copy the presenter's timbre; the Edge TTS voice above is ignored.";
  } else {
    loadVoices();
  }
}
cloneToggle.addEventListener('change', syncCloneControls);

function syncLocalizationMode() {
  const mode = currentMode();
  const speechEnabled = mode !== 'visual';
  const visualEnabled = mode !== 'sound';
  visualOnlyField.value = String(mode === 'visual');
  soundOnlyField.value = String(mode === 'sound');
  if (mode === 'visual') {
    localizeVisualToggle.checked = true;
    visualContextToggle.checked = true;
  } else if (mode === 'sound') {
    localizeVisualToggle.checked = false;
    visualContextToggle.checked = false;
    writeVisualTrackToggle.checked = false;
  }
  setGroupEnabled('[data-speech-control]', speechEnabled);
  setGroupEnabled('[data-visual-control]', visualEnabled);
  submitCopy.textContent = mode === 'visual' ? 'Create visual localization' : mode === 'sound' ? 'Create sound dub' : 'Create localized video';
  updateStepCopy(mode);
  syncCloneControls();
}

modeInputs.forEach(input => input.addEventListener('change', syncLocalizationMode));
syncLocalizationMode();

function setControl(name, value) {
  const control = form.elements.namedItem(name);
  if (!control || value === null || typeof value === 'object') return;
  if (control.type === 'checkbox') control.checked = Boolean(value);
  else control.value = value;
}

async function loadProfiles() {
  try {
    const response = await fetch('/api/profiles');
    if (!response.ok) throw new Error('Profiles unavailable.');
    profiles = (await response.json()).profiles || [];
    profiles.forEach(item => {
      const option = document.createElement('option');
      option.value = item.name;
      option.textContent = `${item.name}${item.built_in ? ' · built in' : ' · saved'}`;
      profilePicker.append(option);
    });
    const preferred = profiles.find(item => item.name === 'quran-id');
    if (preferred) {
      profilePicker.value = preferred.name;
      applyProfile(preferred);
    }
  } catch (error) {
    profileHelp.textContent = error.message;
  }
}

function applyProfile(selected) {
  if (!selected) {
    profileHelp.textContent = 'Using custom form settings.';
    return;
  }
  Object.entries(selected.options).forEach(([name, value]) => setControl(name, value));
  profileHelp.textContent = `${selected.name} loaded · translation uses the local LLM; speech uses the presenter's F5 voice clone.`;
  syncLocalizationMode();
  syncCloneControls();
  document.querySelector('#volume-value').textContent = `${Math.round(volume.value * 100)}%`;
  loadVoices();
}

profilePicker.addEventListener('change', () => {
  const selected = profiles.find(item => item.name === profilePicker.value);
  applyProfile(selected);
});
loadProfiles();

async function loadVoices() {
  if (currentMode() === 'visual') {
    voicePicker.disabled = true;
    voiceHelp.textContent = 'Original soundtrack will be copied without speech synthesis.';
    return;
  }
  if (cloneToggle.checked) {
    voicePicker.disabled = true;
    voiceHelp.textContent = "F5 will copy the presenter's timbre; the Edge TTS voice above is ignored.";
    return;
  }
  const language = targetInput.value.trim();
  if (!language) return;
  voicePicker.disabled = true;
  voicePicker.innerHTML = '<option value="">Loading voices...</option>';
  try {
    const response = await fetch(`/api/voices?language=${encodeURIComponent(language)}`);
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || 'Voice list unavailable.');
    voicePicker.innerHTML = '<option value="">Automatic · male</option>';
    body.voices.forEach(voice => {
      const option = document.createElement('option');
      option.value = voice.name;
      option.textContent = `${voice.name} · ${voice.gender} · ${voice.locale}`;
      voicePicker.append(option);
    });
    voiceHelp.textContent = `${body.voices.length} matching voices · male listed first.`;
  } catch (error) {
    voicePicker.innerHTML = '<option value="">Automatic · male</option>';
    voiceHelp.textContent = error.message;
  } finally {
    voicePicker.disabled = false;
  }
}

targetInput.addEventListener('input', () => {
  clearTimeout(voiceTimer);
  voiceTimer = setTimeout(loadVoices, 500);
});
loadVoices();

const stageRules = [
  ['extract', /extracting|downloading/i], ['transcribe', /transcribing/i],
  ['translate', /translating|model/i], ['voice', /generating.*(?:voice|utterance)|clone voice/i],
  ['create', /synchronizing|creating dubbed|done:/i]
];
const visualStageRules = [
  ['extract', /visual-only mode|downloading/i],
  ['transcribe', /visual context|reading on-screen|slide states/i],
  ['translate', /localizing static|visual translation/i],
  ['voice', /rendering localized visual overlays/i],
  ['create', /done:/i]
];

function estimatedProgress(job, messages) {
  if (job.status === 'completed') return 100;
  const visualOnly = messages.some(message => message.toLowerCase().includes('visual-only mode'));
  if (visualOnly) {
    let visualValue = job.status === 'queued' ? 0 : 2;
    messages.forEach(message => {
      const line = message.toLowerCase();
      if (line.includes('downloading youtube')) visualValue = Math.max(visualValue, 3);
      if (line.includes('visual context: extracting')) visualValue = Math.max(visualValue, 8);
      let match = line.match(/reading on-screen text\s+(\d+)\/(\d+)/);
      if (match) visualValue = Math.max(visualValue, 8 + 22 * Number(match[1]) / Math.max(1, Number(match[2])));
      if (line.includes('slide states')) visualValue = Math.max(visualValue, 32);
      match = line.match(/visual translation batch\s+(\d+)\/(\d+)/);
      if (match) visualValue = Math.max(visualValue, 35 + 35 * Number(match[1]) / Math.max(1, Number(match[2])));
      if (line.includes('rendering localized visual overlays')) visualValue = Math.max(visualValue, 74);
      if (line.startsWith('done:')) visualValue = 100;
    });
    return Math.max(0, Math.min(100, Math.round(visualValue)));
  }
  let value = job.status === 'queued' ? 0 : 1;
  messages.forEach(message => {
    const line = message.toLowerCase();
    if (line.includes('downloading youtube')) value = Math.max(value, 2);
    // A retry keeps its durable history. A new extraction marks a fresh run,
    // so stale percentages from the failed attempt must not win.
    if (line.includes('extracting audio')) value = value > 5 ? 5 : Math.max(value, 5);
    if (line.includes('visual context: extracting')) value = Math.max(value, 8);
    let match = line.match(/reading on-screen text\s+(\d+)\/(\d+)/);
    if (match) value = Math.max(value, 8 + 12 * Number(match[1]) / Math.max(1, Number(match[2])));
    if (line.includes('slide states')) value = Math.max(value, 20);
    if (line.includes('visual track written')) value = Math.max(value, 21);
    if (line.includes('transcribing speech')) value = Math.max(value, 25);
    if (line.includes('grouping ') && line.includes('transcript')) value = Math.max(value, 45);
    if (line.includes('translating ')) value = Math.max(value, 48);
    if (line.includes('preserving ') || line.includes('translation model')) value = Math.max(value, 52);
    if (line.includes('cloning original voice')) value = Math.max(value, 58);
    if (line.includes('generating ') && line.includes('cloned original voice')) value = Math.max(value, 60);
    match = line.match(/clone voice\s+(\d+)\/(\d+)\s+utterances/);
    if (match) value = Math.max(value, 60 + 25 * Number(match[1]) / Math.max(1, Number(match[2])));
    if (line.includes('segment metadata written')) value = Math.max(value, 87);
    if (line.includes('synchronizing speech')) value = Math.max(value, 89);
    if (line.includes('localizing static')) value = Math.max(value, 93);
    if (line.includes('creating dubbed video')) value = Math.max(value, 97);
    if (line.startsWith('done:')) value = 100;
  });
  return Math.max(0, Math.min(100, Math.round(value)));
}

function renderProgress(job, messages) {
  const percent = estimatedProgress(job, messages);
  progressFill.style.width = `${percent}%`;
  progressPercent.textContent = `${percent}%`;
  progressTrack.setAttribute('aria-valuenow', String(percent));
  progressTrack.classList.toggle('failed', job.status === 'failed');
  progressTrack.classList.toggle('interrupted', job.status === 'interrupted');
  progressTrack.classList.toggle('paused', job.status === 'paused');
  progressLabel.textContent = job.status === 'queued' ? 'Queued' : job.status === 'running' ? 'Processing' : job.status === 'completed' ? 'Complete' : job.status === 'failed' ? 'Failed' : job.status === 'interrupted' ? 'Interrupted' : job.status === 'paused' ? 'Paused' : job.status;
}

function renderJob(job) {
  currentJob = job;
  renderTiming(job);
  const messages = job.progress || [];
  const visualJob = messages.some(line => line.toLowerCase().includes('visual-only mode'));
  const soundJob = messages.some(line => line.toLowerCase().includes('sound-only mode'));
  if (visualJob) updateStepCopy('visual'); else if (soundJob) updateStepCopy('sound');
  renderProgress(job, messages);
  consoleBox.innerHTML = messages.length ? messages.map(line => `<span>${escapeHtml(line)}</span>`).join('') : '<span>Job queued. Waiting for the processing slot...</span>';
  consoleBox.scrollTop = consoleBox.scrollHeight;
  let active = -1;
  const rules = visualJob ? visualStageRules : stageRules;
  messages.forEach(line => rules.forEach(([, rule], index) => { if (rule.test(line)) active = Math.max(active, index); }));
  document.querySelectorAll('#steps li').forEach((step, index) => {
    step.classList.toggle('done', job.status === 'completed' || index < active);
    step.classList.toggle('active', job.status === 'running' && index === active);
  });
  caption.textContent = job.status === 'queued' ? 'Queued — waiting for a free processing slot...' : job.status === 'running' ? `Working · stage ${Math.max(1, active + 1)} of 5` : job.status === 'paused' ? (job.pause_pending ? 'Pausing — finishing the current stage...' : 'Paused at a checkpoint — the queue is holding until you resume.') : job.status === 'completed' ? 'Everything is synchronized.' : 'Processing stopped.';
  processCard.classList.toggle('running', job.status === 'running');
  retryButton.classList.toggle('hidden', !['failed', 'interrupted', 'cancelled'].includes(job.status));
  const pausable = ['queued', 'running', 'paused'].includes(job.status);
  pauseButton.classList.toggle('hidden', !pausable);
  discardButton.classList.toggle('hidden', job.status !== 'paused');
  if (job.status === 'paused') {
    pauseButton.textContent = job.pause_pending ? 'Stopping current stage…' : 'Resume job';
    pauseButton.disabled = Boolean(job.pause_pending);
  } else if (pausable) {
    pauseButton.textContent = 'Pause job';
    pauseButton.disabled = false;
  }
  if (job.status === 'completed') {
    clearTimeout(pollTimer); submit.disabled = false; result.classList.remove('hidden');
    document.querySelector('#video-link').href = job.video_url;
    const subtitle = document.querySelector('#subtitle-link');
    subtitle.classList.toggle('hidden', !job.subtitle_url); subtitle.href = job.subtitle_url || '#';
  } else if (job.status === 'failed' || job.status === 'interrupted' || job.status === 'cancelled') {
    clearTimeout(pollTimer); submit.disabled = false;
    errorBox.textContent = job.error || (job.status === 'cancelled' ? 'Job discarded.' : 'The job failed.');
  }
}

retryButton.addEventListener('click', async () => {
  if (!currentJob) return;
  retryButton.disabled = true; errorBox.textContent = '';
  try {
    const response = await fetch(`/api/jobs/${currentJob.id}/retry`, {method: 'POST'});
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || 'Could not retry the job.');
    renderJob(body); poll(`/api/jobs/${body.id}`);
  } catch (error) {
    errorBox.textContent = error.message;
  } finally {
    retryButton.disabled = false;
  }
});

async function poll(url) {
  clearTimeout(pollTimer);
  try {
    const response = await fetch(url);
    if (!response.ok) throw new Error('Could not read job status.');
    const job = await response.json(); renderJob(job);
    if (job.status === 'queued' || job.status === 'running' || job.status === 'paused') pollTimer = setTimeout(() => poll(url), 1200);
  } catch (error) {
    errorBox.textContent = error.message; submit.disabled = false;
  }
}

pauseButton.addEventListener('click', async () => {
  if (!currentJob) return;
  const action = currentJob.status === 'paused' ? 'resume' : 'pause';
  pauseButton.disabled = true;
  try {
    const response = await fetch(`/api/jobs/${currentJob.id}/${action}`, {method: 'POST'});
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || `Could not ${action} the job.`);
    errorBox.textContent = '';
    renderJob(body); poll(`/api/jobs/${body.id}`);
  } catch (error) {
    errorBox.textContent = error.message;
    pauseButton.disabled = false;
  }
});

discardButton.addEventListener('click', async () => {
  if (!currentJob) return;
  discardButton.disabled = true;
  try {
    const response = await fetch(`/api/jobs/${currentJob.id}/cancel`, {method: 'POST'});
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || 'Could not discard the job.');
    errorBox.textContent = '';
    renderJob(body);
  } catch (error) {
    errorBox.textContent = error.message;
  } finally {
    discardButton.disabled = false;
  }
});

async function resumeLatestJob() {
  try {
    const response = await fetch('/api/jobs');
    if (!response.ok) return;
    const body = await response.json();
    const jobs = body.jobs || [];
    if (!jobs.length) {
      currentJob = undefined;
      clearTimeout(pollTimer);
      caption.textContent = 'Waiting for a video.';
      consoleBox.innerHTML = '<span>voicecover is ready.</span>';
      progressFill.style.width = '0%';
      progressPercent.textContent = '0%';
      progressLabel.textContent = 'Waiting';
      processCard.classList.remove('running');
      submit.disabled = false;
      result.classList.add('hidden');
      errorBox.textContent = '';
      retryButton.classList.add('hidden');
      pauseButton.classList.add('hidden');
      discardButton.classList.add('hidden');
      document.querySelectorAll('#steps li').forEach(step => step.classList.remove('done', 'active'));
      return;
    }
    const active = [...jobs].reverse().find(job => ['running', 'queued', 'paused'].includes(job.status));
    const job = active || jobs[jobs.length - 1];
    renderJob(job);
    if (active) {
      submit.disabled = true;
      poll(`/api/jobs/${job.id}`);
    }
  } catch (_) {
    // The page remains usable when job history is temporarily unavailable.
  }
}

form.addEventListener('submit', async event => {
  event.preventDefault(); errorBox.textContent = ''; result.classList.add('hidden');
  if (sourceMode === 'upload' && !fileInput.files[0]) { errorBox.textContent = 'Choose a video first.'; return; }
  if (sourceMode === 'youtube' && !youtubeInput.value.trim()) { errorBox.textContent = 'Enter a YouTube URL first.'; return; }
  const data = new FormData(form);
  const mode = currentMode();
  data.delete('localization_mode');
  data.set('visual_only', String(mode === 'visual'));
  data.set('sound_only', String(mode === 'sound'));
  if (!data.has('write_subtitles')) data.set('write_subtitles', 'false');
  if (!data.has('install_models')) data.set('install_models', 'false');
  if (!data.has('preserve_arabic')) data.set('preserve_arabic', 'false');
  if (!data.has('natural_speech')) data.set('natural_speech', 'false');
  if (!data.has('visual_context')) data.set('visual_context', 'false');
  if (!data.has('write_visual_track')) data.set('write_visual_track', 'false');
  if (!data.has('localize_visual_text')) data.set('localize_visual_text', 'false');
  if (!data.has('visual_only')) data.set('visual_only', 'false');
  if (!data.has('sound_only')) data.set('sound_only', 'false');
  if (!data.has('voice_clone')) data.set('voice_clone', 'false');
  if (!data.has('write_segments')) data.set('write_segments', 'false');
  if (!data.has('write_bilingual')) data.set('write_bilingual', 'false');
  submit.disabled = true; caption.textContent = 'Uploading and preparing your job...'; processCard.classList.add('running');
  consoleBox.innerHTML = '<span>Sending source to voicecover...</span>';
  try {
    const response = await fetch('/api/jobs', {method: 'POST', body: data});
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || 'Could not start the job.');
    const job = body.jobs ? body.jobs[0] : body;
    renderJob(job); poll(`/api/jobs/${job.id}`); loadJobList(true);
  } catch (error) {
    submit.disabled = false; processCard.classList.remove('running'); errorBox.textContent = error.message;
  }
});

function escapeHtml(value) {
  const node = document.createElement('div'); node.textContent = value; return node.innerHTML;
}

// ── job vault ─────────────────────────────────────────────────────

const jobList = document.querySelector('#job-list');

function actionButton(action, id, label) {
  return `<button type="button" data-action="${action}" data-id="${escapeHtml(id)}">${label}</button>`;
}

function renderJobList(jobs) {
  if (!jobs.length) {
    jobList.innerHTML = '<p class="jobs-empty">No jobs yet — submit a video above.</p>';
    return;
  }
  jobList.innerHTML = [...jobs].reverse().map(job => {
    const created = job.created_at ? new Date(job.created_at) : null;
    const when = created && !Number.isNaN(created.getTime())
      ? created.toLocaleString([], {month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'})
      : '—';
    const actions = [];
    if (['queued', 'running'].includes(job.status)) actions.push(actionButton('pause', job.id, 'Pause'));
    if (job.status === 'paused') actions.push(actionButton('resume', job.id, 'Resume'), actionButton('cancel', job.id, 'Discard'));
    if (['failed', 'interrupted', 'cancelled'].includes(job.status)) actions.push(actionButton('retry', job.id, 'Retry'));
    if (job.status === 'completed' && job.video_url) {
      actions.push(`<a href="${job.video_url}">Download</a>`, actionButton('retry', job.id, 'Reprocess'));
    }
    if (job.status !== 'running') actions.push(actionButton('delete', job.id, 'Delete'));
    actions.push(actionButton('log', job.id, 'Log'));
    return `<div class="job-row" data-id="${escapeHtml(job.id)}">
      <div class="job-row-main" title="Show this job in the processing room">
        <strong>${escapeHtml(job.source || job.name || job.id)}</strong>
        <small>${when} · attempt ${job.attempts ?? 1}</small>
      </div>
      <span class="job-chip ${escapeHtml(job.status)}">${escapeHtml(job.status)}</span>
      <div class="job-row-actions">${actions.join('')}</div>
    </div>`;
  }).join('');
}

async function loadJobList(force = false) {
  // An open log or an in-flight action owns the DOM until it finishes.
  if (!force && (jobList.querySelector('.job-log') || jobList.querySelector('button:disabled'))) return;
  try {
    const response = await fetch('/api/jobs');
    if (!response.ok) return;
    const body = await response.json();
    renderJobList(body.jobs || []);
  } catch (_) {
    // The list reappears on the next tick when the server is briefly away.
  }
}

async function toggleJobLog(control, id) {
  const row = control.closest('.job-row');
  const open = row.querySelector('.job-log');
  if (open) {
    open.remove();
    control.textContent = 'Log';
    loadJobList(true);
    return;
  }
  control.disabled = true;
  try {
    const response = await fetch(`/api/jobs/${id}/log`);
    if (!response.ok) throw new Error('Log unavailable.');
    const body = await response.json();
    const lines = (body.lines && body.lines.length)
      ? body.lines
      : (body.progress || []).map(line => `· ${line}`);
    const pre = document.createElement('pre');
    pre.className = 'job-log';
    pre.textContent = lines.length ? lines.join('\n') : 'No log entries yet.';
    row.append(pre);
    control.textContent = 'Hide log';
  } catch (error) {
    errorBox.textContent = error.message;
  } finally {
    control.disabled = false;
  }
}

async function watchJob(id) {
  try {
    const response = await fetch(`/api/jobs/${id}`);
    if (!response.ok) return;
    const job = await response.json();
    clearTimeout(pollTimer);
    result.classList.add('hidden');
    errorBox.textContent = '';
    renderJob(job);
    if (['queued', 'running', 'paused'].includes(job.status)) {
      submit.disabled = true;
      poll(`/api/jobs/${id}`);
    } else {
      submit.disabled = false;
    }
  } catch (_) {
    // Watching is best-effort; the vault list stays authoritative.
  }
}

jobList.addEventListener('click', async event => {
  const control = event.target.closest('[data-action]');
  if (control) {
    const action = control.dataset.action;
    const id = control.dataset.id;
    if (action === 'log') { toggleJobLog(control, id); return; }
    if (action === 'delete' && !window.confirm('Delete this job and all of its files?')) return;
    control.disabled = true;
    try {
      const response = await fetch(
        action === 'delete' ? `/api/jobs/${id}` : `/api/jobs/${id}/${action}`,
        {method: action === 'delete' ? 'DELETE' : 'POST'}
      );
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail || `Could not ${action} the job.`);
      }
      errorBox.textContent = '';
      control.disabled = false;
      loadJobList(true);
      if (currentJob && currentJob.id === id) {
        if (action === 'delete') {
          clearTimeout(pollTimer);
          resumeLatestJob();
        } else {
          watchJob(id);
        }
      }
    } catch (error) {
      control.disabled = false;
      errorBox.textContent = error.message;
    }
    return;
  }
  const main = event.target.closest('.job-row-main');
  if (main) {
    const row = main.closest('.job-row');
    if (row) watchJob(row.dataset.id);
  }
});

setInterval(loadJobList, 3000);
loadJobList();

resumeLatestJob();
