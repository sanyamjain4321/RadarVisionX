// ═══════════════════════════════════════════════════════════════════
// UI — SAR Ship Detector (YOLO + Auto-reconnect + Metrics Fullscreen)
// ═══════════════════════════════════════════════════════════════════

// ── YOLO State Machine ────────────────────────────────────────────
// States: checking | no_model | training | loading | ready | error
let _trainingPollTimer = null;

/** Listen for state changes fired by yolo.js */
document.addEventListener('yolo-state', e => {
  const { state, msg, isSAR, modelType } = e.detail;
  updateYOLOStatus(state, msg, modelType, isSAR);
});

/** Auto-reconnect: backend came back online */
document.addEventListener('yolo-reconnected', async () => {
  log('✓ Backend reconnected — re-initialising YOLO pipeline…', 'hit');
  
  // Force full pipeline as requested
  const psel = G('det-pipeline');
  if(psel) psel.value = 'hybrid';
  onPipelineChange('hybrid');
  
  const msel = G('yolo-mode-sel');
  if(msel) msel.value = 'backend';
  onYOLOModeChange('backend');
  
  await initYOLOPipeline();
});

function closeWelcomeScreen() {
  const ws = G('welcome-screen');
  if(ws) ws.classList.add('fade-out');
}

/** Backend permanently offline → simulate mode activated */
document.addEventListener('yolo-fallback-simulate', () => {
  const badge = G('yolo-backend-badge');
  if(badge){
    badge.className   = 'backend-badge offline';
    badge.textContent = '⚠ Simulate Mode';
    badge.style.borderColor = 'var(--amber)';
    badge.style.color       = 'var(--amber)';
  }
  log('⚠ Backend offline — YOLO simulate mode active (CFAR-only passthrough)', 'miss');
  log('  → Start backend: start_backend.bat (will auto-reconnect)', 'ai');
  // Enable run button in simulate mode so user can still use CFAR
  const pipeline = getDetPipeline();
  if(CFAR && CFAR.isArmed() && pipeline !== 'yolo_backend'){
    const runBtn = G('run-btn');
    if(runBtn) runBtn.disabled = false;
  }
});

function updateYOLOStatus(state, msg, modelType, isSAR){
  const badge     = G('yolo-badge');
  const modelDisp = G('yolo-model-name');
  const stateDisp = G('yolo-state-disp');
  const trainPnl  = G('yolo-train-panel');
  const trainBtn  = G('yolo-train-btn');
  const cancelBtn = G('yolo-cancel-btn');
  const loadBtn   = G('yolo-load-btn');
  const runBtn    = G('run-btn');

  // Badge
  const badgeCfg = {
    checking:  { text:'CHECKING', cls:'opt-off' },
    no_model:  { text:'NO MODEL', cls:'opt-warn' },
    training:  { text:'TRAINING', cls:'opt-train' },
    loading:   { text:'LOADING',  cls:'opt-off' },
    ready:     { text:'READY',    cls:'opt-on' },
    error:     { text:'ERROR',    cls:'opt-err' },
  };
  const b = badgeCfg[state] || badgeCfg.error;
  if(badge){ badge.textContent = b.text; badge.className = 'opt-badge ' + b.cls; }
  if(stateDisp) stateDisp.textContent = msg || state;
  if(modelDisp){
    modelDisp.textContent = modelType || '—';
    modelDisp.style.color = isSAR ? 'var(--green)' : 'var(--amber)';
  }

  // Training panel visibility
  if(trainPnl){
    trainPnl.style.display = (state === 'no_model' || state === 'training' || state === 'error') ? 'block' : 'none';
  }
  if(trainBtn)  trainBtn.style.display  = (state === 'no_model' || state === 'error') ? 'block' : 'none';
  if(cancelBtn) cancelBtn.style.display = (state === 'training')                       ? 'block' : 'none';

  // Load ONNX button only in onnx mode
  const yoloMode = G('yolo-mode-sel');
  if(loadBtn) loadBtn.style.display = (yoloMode && yoloMode.value === 'onnx') ? 'block' : 'none';

  // Run button: enabled as long as CFAR is armed.
  // YOLO state does NOT block the run button — CFAR always works standalone.
  const cfArmed = typeof CFAR !== 'undefined' && CFAR.isArmed();
  if(runBtn){
    // Only disable if CFAR is not yet armed. YOLO state is informational only.
    runBtn.disabled = !cfArmed;
  }

  // Engine display
  const engEl = G('ov-engine');
  if(engEl){
    if(state === 'ready' && isSAR){
      engEl.textContent = 'CFAR→YOLO(SAR)→AI';
      engEl.style.color = 'var(--green)';
    } else if(state === 'ready'){
      engEl.textContent = 'CFAR→YOLO(COCO)→AI';
      engEl.style.color = 'var(--amber)';
    } else if(state === 'training'){
      engEl.textContent = 'Training YOLO…';
      engEl.style.color = 'var(--blue)';
    } else {
      engEl.textContent = 'CFAR + AI-LR';
      engEl.style.color = 'var(--purple)';
    }
  }
}

/** Auto-detect best YOLO mode on page load. */
async function initYOLOPipeline(){
  log('⟳ Initialising YOLO pipeline…', 'ai');
  updateYOLOStatus('checking', 'Connecting to YOLO backend...');
  const wsText   = G('ws-text');
  if(wsText) wsText.textContent = 'Connecting to YOLO backend...';

  // Force Backend Defaults visually
  const psel = G('det-pipeline');
  if(psel) psel.value = 'hybrid';
  onPipelineChange('hybrid');
  
  const msel = G('yolo-mode-sel');
  if(msel) msel.value = 'backend';
  onYOLOModeChange('backend');

  // 1. Try backend
  const r = await YOLOInference.pingBackend();
  if(r.ok){
    updateYOLOModel_DisplayFromHealth(r.info);
    if(YOLOInference.isSARModel()){
      log(`✓ Backend online — SAR model loaded (${YOLOInference.getModelType()})`, 'hit');
      if(wsText) wsText.textContent = 'Backend online ✓';
      setTimeout(closeWelcomeScreen, 1200);
    } else {
      log(`⚠ Backend online but no SAR model — ${YOLOInference.getModelType()}`, 'miss');
      log('  → Run start_all.bat to auto-train, or click "Start Training"', 'ai');
      if(wsText) wsText.textContent = 'Backend online ✓ (No SAR Model)';
      setTimeout(closeWelcomeScreen, 1500);
    }
    return;
  }

  // 2. Backend offline -> No ONNX fallback by default (simulate mode as safety net)
  log('⬡ Backend offline — auto-reconnect active (every 2s)…', 'miss');
  updateYOLOStatus('error', 'Backend offline — Simulate mode active');
  if(wsText) wsText.textContent = 'Backend unavailable — using simulation';
  
  YOLOInference.setMode('simulate'); // Manually trigger simulate locally while reconnect polling
  YOLOInference.startReconnectLoop();
  
  setTimeout(closeWelcomeScreen, 2000);
}

function updateYOLOModel_DisplayFromHealth(healthJson){
  const backBadge = G('yolo-backend-badge');
  const backDisp  = G('yolo-back-disp');
  if(backBadge){
    backBadge.className   = 'backend-badge online';
    backBadge.textContent = '✓ Backend Online';
  }
  if(backDisp){
    backDisp.textContent  = 'Online';
    backDisp.style.color  = 'var(--green)';
  }
}



// ── Tab switching (Metrics = fullscreen, SAR = sidebar visible) ───
function switchTab(t){
  ['sar','realsar','metrics'].forEach(id => {
    const el = G('tab-'+id);
    if(el) el.style.display = (id===t) ? (id==='sar' ? 'block' : 'block') : 'none';
  });
  document.querySelectorAll('.tab').forEach((el,i) => {
    el.classList.toggle('active', ['sar','realsar','metrics'][i]===t);
  });

  // Toggle sidebar: show in SAR/RealSAR mode, hide in Metrics mode
  const sidebar = G('sidebar');
  const mainDiv = G('main');
  if(t === 'metrics'){
    if(sidebar)  sidebar.style.display = 'none';
    if(mainDiv)  mainDiv.classList.add('metrics-active');
    // Expand metrics to full screen
    const metricsEl = G('tab-metrics');
    if(metricsEl){
      metricsEl.style.display = 'flex';
      metricsEl.classList.add('metrics-fullscreen');
    }
    setTimeout(() => { Met.redraw(); drawLearningCurve(); }, 80);
  } else {
    if(sidebar)  sidebar.style.display = '';
    if(mainDiv)  mainDiv.classList.remove('metrics-active');
    const metricsEl = G('tab-metrics');
    if(metricsEl) metricsEl.classList.remove('metrics-fullscreen');
  }
}

// ── Pipeline helpers ───────────────────────────────────────────────
function getDetPipeline(){
  const el = G('det-pipeline');
  return el ? el.value : 'cfar';
}

function onPipelineChange(val){
  const yoloCtrl = G('yolo-controls');
  const badge    = G('yolo-badge');
  const pipeDisp = G('yolo-pipeline-disp');
  const engEl    = G('ov-engine');

  if(val === 'cfar'){
    if(yoloCtrl) yoloCtrl.style.display = 'none';
    if(pipeDisp) pipeDisp.textContent = 'CFAR+AI';
    if(engEl){ engEl.textContent = 'CFAR + AI-LR'; engEl.style.color = 'var(--purple)'; }
    updateRunBtn('cfar');
  } else {
    if(yoloCtrl) yoloCtrl.style.display = 'block';
    const lbl = val === 'hybrid' ? 'CFAR→YOLO→AI' : 'YOLO Backend';
    if(pipeDisp) pipeDisp.textContent = lbl;
    const msel = G('yolo-mode-sel');
    onYOLOModeChange(msel ? msel.value : 'backend');
    updateRunBtn(val);
  }
  log(`Pipeline: ${val === 'cfar' ? 'CFAR Only' : val === 'hybrid' ? 'CFAR→YOLO→AI' : 'YOLO Backend'}`, 'info');
}

function onYOLOModeChange(mode){
  YOLOInference.setMode(mode);
  const urlRow   = G('backend-url-row');
  const modeDisp = G('yolo-mode-disp');
  const loadBtn  = G('yolo-load-btn');
  if(urlRow)   urlRow.style.display   = (mode === 'backend') ? 'flex'  : 'none';
  if(loadBtn)  loadBtn.style.display  = (mode === 'onnx')    ? 'block' : 'none';
  if(modeDisp) modeDisp.textContent   = { onnx: 'ONNX (browser)', backend: 'FastAPI Backend' }[mode] || mode;
  log(`YOLO mode: ${mode}`, 'ai');
}

function updateRunBtn(pipeline){
  const btn = G('run-btn');
  if(!btn) return;
  btn.textContent = {
    cfar:         '▶ Run CFAR + AI Detection',
    hybrid:       '▶ Run CFAR → YOLO → AI',
    yolo_backend: '▶ Run YOLO Backend',
  }[pipeline] || '▶ Run CFAR + AI Detection';
}

// ── Load ONNX model (manual) ───────────────────────────────────────
async function loadYOLOModel(){
  const prog = G('yolo-prog'), bar = G('yolo-prog-bar'), txt = G('yolo-prog-txt');
  const btn  = G('yolo-load-btn');
  if(prog) prog.style.display = 'block';
  if(btn)  btn.disabled = true;
  updateYOLOStatus('loading', 'Loading ONNX model…');
  await YOLOInference.loadONNXModel((msg, pct) => {
    if(txt) txt.textContent = msg;
    if(bar) bar.style.width = (pct < 0 ? 100 : pct) + '%';
    if(pct < 0 && bar) bar.style.background = 'var(--red)';
    log(msg, pct < 0 ? 'miss' : 'ai');
  });
  setTimeout(() => { if(prog) prog.style.display='none'; if(btn) btn.disabled=false; }, 1500);
}

// ── Backend ping ───────────────────────────────────────────────────
async function doBackendPing(){
  const backBadge = G('yolo-backend-badge');
  const backDisp  = G('yolo-back-disp');
  const urlEl     = G('backend-url');
  if(urlEl) YOLOInference.setBackendUrl(urlEl.value.trim());

  const { ok, info } = await YOLOInference.pingBackend();

  if(backBadge){
    backBadge.className   = 'backend-badge ' + (ok ? 'online' : 'offline');
    backBadge.textContent = ok ? '✓ Backend Online' : '⬡ Backend Offline';
  }
  if(backDisp){
    backDisp.textContent = ok ? 'Online' : 'Offline';
    backDisp.style.color = ok ? 'var(--green)' : 'var(--red)';
  }

  if(ok && info && info.model){
    const m = info.model;
    const modelNameEl = G('yolo-model-name');
    if(modelNameEl){
      modelNameEl.textContent = m.model_type || '—';
      modelNameEl.style.color = m.is_sar ? 'var(--green)' : 'var(--amber)';
    }
    log(`Backend ✓ — ${m.model_type} (${m.name || '?'})`, 'hit');
  } else {
    log('Backend offline — start_backend.bat or start_all.bat', 'miss');
  }
}

// ── Arm Detector ───────────────────────────────────────────────────
function armDetector(){
  const btn = G('arm-btn'), st = G('model-status');
  btn.disabled = true;
  st.textContent = '⟳ Arming…';
  log('Arming CFAR + AI pipeline…', 'info');
  if(typeof CfarDebug !== 'undefined') CfarDebug.log('Arming CFAR pipeline…');
  CFAR.arm(msg => {
    log(msg, 'info');
    if(CFAR.isArmed()){
      st.textContent  = '✓ Armed';
      btn.textContent = '✓ Armed';
      log('━━ CFAR pipeline armed ━━', 'hit');
      if(typeof CfarDebug !== 'undefined') CfarDebug.ok('CFAR pipeline armed ✓');
      updateAIDisplay();
      // ALWAYS enable Run button once CFAR is armed — YOLO state is non-blocking
      const runBtn = G('run-btn');
      if(runBtn) runBtn.disabled = false;
      const yState   = YOLOInference.getState();
      const pipeline = getDetPipeline();
      if(pipeline === 'cfar'){
        log('▶ Ready — CFAR Only mode', 'hit');
      } else if(yState === 'ready'){
        log('▶ Ready — CFAR → YOLO → AI pipeline fully online', 'hit');
      } else {
        log('▶ Ready — running CFAR-only (YOLO unavailable). Start backend for full pipeline.', 'ai');
        if(typeof CfarDebug !== 'undefined') CfarDebug.warn('YOLO offline — CFAR-only fallback active');
      }
    }
  });
}

// ── Learning curve ─────────────────────────────────────────────────
function drawLearningCurve(){
  try {
    const curve = AIModel.getCurve();
    if(!curve || !curve.loss || !curve.loss.length) return;
    const cv = G('aisc'); if(!cv) return;
    const c = cv.getContext('2d'), W = cv.clientWidth||860, H = cv.clientHeight||110;
    cv.width = W; cv.height = H;
    c.fillStyle = '#ffffff'; c.fillRect(0,0,W,H);
    const n = curve.loss.length; if(n<2) return;
    
    // Grid and Axes
    c.strokeStyle = 'rgba(0,0,0,0.08)'; c.lineWidth = 1;
    for(let i=0;i<=4;i++){ const y=H-H*i/4; c.beginPath(); c.moveTo(0,y); c.lineTo(W,y); c.stroke(); }
    c.strokeStyle = 'rgba(0,0,0,0.2)'; c.lineWidth = 2;
    c.beginPath(); c.moveTo(0,H); c.lineTo(W,H); c.moveTo(0,0); c.lineTo(0,H); c.stroke();

    const maxLoss = Math.max(...curve.loss)*1.2||1;
    const series  = [
      {d:curve.f1,                          c:'#f59e0b', label:'F1 Score'},
      {d:curve.acc,                         c:'#7c3aed', label:'Accuracy'},
      {d:curve.loss.map(v=>v/maxLoss*100),  c:'#ef4444', label:'Loss'},
    ];
    series.forEach(s => {
      if(s.d.length<2) return;
      // Draw direct point-to-point line (Tension: 0)
      c.strokeStyle=s.c; c.lineWidth=2.5; c.beginPath();
      const points = s.d.map((v,i) => ({x:i/(n-1||1)*W, y:H-Number(v)/100*H}));
      if(points.length>0){
        c.moveTo(points[0].x, points[0].y);
        for(let i=1; i<points.length; i++){
          c.lineTo(points[i].x, points[i].y);
        }
      }
      c.stroke();

      // Anchored dots for perfect alignment
      points.forEach(p=>{
        c.fillStyle='#ffffff'; c.beginPath(); c.arc(p.x, p.y, 4, 0, Math.PI*2); c.fill();
        c.strokeStyle=s.c; c.lineWidth=1.5; c.beginPath(); c.arc(p.x, p.y, 3.5, 0, Math.PI*2); c.stroke();
      });
    });
    c.font='bold 10px Inter,sans-serif'; c.textBaseline='top';
    series.forEach((s,i) => { c.fillStyle=s.c; c.fillText(s.label, 12+i*70, 6); });
  } catch (e) {
    console.error("AIModel curve error:", e);
  }
}

// ── AI display ─────────────────────────────────────────────────────
function updateAIDisplay(){
  try {
    const on  = AIModel.isEnabled();
    G('ai-badge').textContent = on ? 'ON' : 'OFF';
    G('ai-badge').className   = 'opt-badge ' + (on?'opt-on':'opt-off');
    G('ai-mode-display').textContent = on ? 'Inference' : 'Bypassed';
    G('ai-epochs').textContent = AIModel.getEpochs();
    const acc = AIModel.getAccuracy();
    G('ai-acc').textContent = acc != null ? (acc*100).toFixed(1)+'%' : '—';
    const learnBtn = G('learn-btn');
    if(learnBtn){
      const {pos,neg} = AIModel.getSampleBreakdown();
      const total = pos+neg;
      learnBtn.textContent = total === 0
        ? '⟳ Learn — run detection first'
        : `⟳ Learn +10 epochs  [${pos}✓  ${neg}✗  total:${total}]`;
      learnBtn.style.opacity = total === 0 ? '0.5' : '1';
    }
  } catch (e) {
    console.error("AIModel display error:", e);
    const learnBtn = G('learn-btn');
    if(learnBtn) {
      learnBtn.textContent = '⟳ Learn (Offline)';
      learnBtn.style.opacity = '0.5';
    }
  }
}

// ── Progress UI Helper ─────────────────────────────────────────────
window.updateLoadingUI = function(msg) {
  const pw = G('pw'), pl = G('pl');
  if (pw && pl) {
    pw.style.display = 'block';
    pl.textContent = msg;
  }
};

// ── Run Detection ──────────────────────────────────────────────────
async function runDetection(){
  if(!CFAR.isArmed()){
    log('⚠ Detector not armed — auto-arming now…', 'miss');
    // AUTO-ARM: don't make the user click a separate button
    armDetector();
    // Wait for arming to complete, then retry
    setTimeout(() => { if(CFAR.isArmed()) runDetection(); }, 600);
    return;
  }

  const pipeline = getDetPipeline();

  const runBtn = G('run-btn');
  runBtn.disabled = true;
  runBtn.textContent = '⟳ Running…';

  const pw=G('pw'), pf=G('pf'), pl=G('pl');
  pw.style.display='block'; pf.style.width='8%';
  pl.textContent='Extracting pixels…';
  G('ov-st').textContent='detecting…';

  const scName = G('ov-sc') ? G('ov-sc').textContent : '';

  let pfa    = +G('pfa').value * 1e-4;
  let winR   = Math.round(+G('win').value / 2);
  let minArea= +G('area').value;
  let iouTh  = +G('iou').value;
  const aiTh   = +G('aith').value;

  // Task 5: Boost Recall (relax thresholds intentionally to feed more candidates to AI)
  pfa = Math.min(pfa * 2.0, 5e-3);
  minArea = Math.max(3, minArea - 2);

  // Task 7: Auto Parameter Tuning based on scenario
  if (scName.includes('Rough Sea') || scName.includes('Coast')) {
      // Clutter-heavy: slightly tighten so pipeline doesn't choke out 
      pfa = pfa * 0.5;
      minArea = minArea + 2;
      iouTh -= 0.10; // Stricter NMS merging
  } else if (scName.includes('Stealth')) {
      // Very faint targets: aggressive relax
      pfa = pfa * 3.0;
  }

  let tCFAR=0, tYOLO=0, tAI=0;
  let yoloDets=null, yoloRejected=[];

  // Reset debug panel
  if(typeof CfarDebug !== 'undefined') CfarDebug.updateCounts(null, null, null, null);

  try {
    const t0     = performance.now();
    const canvas = Scene.getCanvas();

    // ── FAILSAFE: verify canvas is valid before running CFAR ────
    if(!canvas || canvas.width < 10 || canvas.height < 10){
      log('⚠ Canvas not ready — regenerating scene…', 'miss');
      if(typeof CfarDebug !== 'undefined') CfarDebug.warn('Canvas invalid — regenerating…');
      Scene.gen();
      await new Promise(r=>setTimeout(r,100));
    }

    // Step 1: CFAR
    pf.style.width='25%'; pl.textContent='Running OS-CFAR…';
    await new Promise(r=>setTimeout(r,10));

    let det;
    try {
      det = await CFAR.detect(canvas, { pfa, winR, minArea, iouTh });
    } catch(cfarErr) {
      // FAILSAFE: if CFAR crashes, regenerate scene and try once more
      console.error('[CFAR] Crash:', cfarErr);
      log('⚠ CFAR crashed — regenerating scene and retrying…', 'miss');
      if(typeof CfarDebug !== 'undefined') CfarDebug.error('CFAR crash: ' + cfarErr.message);
      Scene.gen();
      await new Promise(r=>setTimeout(r,200));
      try {
        det = await CFAR.detect(Scene.getCanvas(), { pfa, winR, minArea, iouTh });
      } catch(e2) {
        log('❌ CFAR failed after retry: ' + e2.message, 'miss');
        if(typeof CfarDebug !== 'undefined') CfarDebug.error('CFAR retry failed: ' + e2.message);
        det = { dets:[], ms:0, error: e2.message };
      }
    }

    tCFAR        = det.ms;
    let cfarDets = det.dets;

    // Guard: if canvas was blank, det.error might be set
    if(det.error){
      log('❌ CFAR: ' + det.error, 'miss');
      if(typeof CfarDebug !== 'undefined') CfarDebug.updateCounts(0, null, null, det.error);
      // Try regenerating scene as a last resort
      log('⟳ Regenerating scene…', 'info');
      Scene.gen();
      G('ov-st').textContent = 'error — scene regenerated, try again';
      pf.style.width='100%';
      await new Promise(r=>setTimeout(r,300));
      pw.style.display='none';
      runBtn.disabled = false;
      updateRunBtn(pipeline);
      return;
    }

    pf.style.width='45%'; pl.textContent=`CFAR: ${cfarDets.length} candidates…`;
    await new Promise(r=>setTimeout(r,10));

    // Update debug panel with CFAR count
    if(typeof CfarDebug !== 'undefined') CfarDebug.updateCounts(cfarDets.length, null, null, null);

    if(cfarDets.length === 0){
      log('⚠ CFAR found 0 detections. Try: lower PFA slider, New SAR Scene, or smaller Min Area.', 'miss');
      if(typeof CfarDebug !== 'undefined') CfarDebug.warn('0 detections — adjust parameters');
    }

    // Step 2: YOLO (hybrid / backend / simulate pipelines)
    if(pipeline === 'hybrid' || pipeline === 'yolo_backend'){
      // YOLO ready check — fallback to simulate if not ready
      if(!YOLOInference.isReady()){
        const st = YOLOInference.getState();
        if(st === 'training'){
          log('⚠ YOLO is training — running CFAR-only this round', 'miss');
        } else {
          log(`⚠ YOLO not ready (${st}) — falling back to CFAR-only`, 'miss');
        }
        if(typeof CfarDebug !== 'undefined') CfarDebug.warn('YOLO unavailable — CFAR-only');
        yoloDets = null;
        yoloRejected = [];
      } else {
        pf.style.width='55%'; pl.textContent=`YOLO: processing ${cfarDets.length} ROIs…`;
        await new Promise(r=>setTimeout(r,10));
        try {
          const yoloResult  = await YOLOInference.refineCFARDets(canvas, cfarDets);
          tYOLO             = yoloResult.ms;
          yoloDets          = yoloResult.refined;
          yoloRejected      = yoloResult.rejected || [];
          pf.style.width='68%'; pl.textContent=`YOLO: ${yoloDets.length} validated…`;
          await new Promise(r=>setTimeout(r,10));
          if(typeof CfarDebug !== 'undefined') CfarDebug.updateCounts(cfarDets.length, yoloDets.length, null, null);
        } catch(yoloErr){
          log(`⚠ YOLO failed (${yoloErr.message}) — using CFAR-only results`, 'miss');
          if(typeof CfarDebug !== 'undefined') CfarDebug.error('YOLO: ' + yoloErr.message);
          yoloDets = null; yoloRejected = [];
        }
      }
    }

    // Step 3: AI filter
    const detsForAI = yoloDets !== null ? yoloDets : cfarDets;
    pf.style.width='82%'; pl.textContent=`AI filter on ${detsForAI.length} dets…`;
    await new Promise(r=>setTimeout(r,10));
    const aiT0 = performance.now();
    const { W, H } = Scene.getWH();
    let kept = detsForAI;
    let filtered = [];
    try {
      const aiResult = AIModel.filter(detsForAI, aiTh, W, H);
      kept = aiResult.kept || detsForAI;
      filtered = aiResult.filtered || [];
    } catch (e) {
      console.error("AIModel filter error:", e);
      kept = detsForAI;
      filtered = [];
      log('⚠ AI Filter fail, using raw detections', 'miss');
    }
    tAI = +(performance.now()-aiT0).toFixed(1);
    pf.style.width='92%'; pl.textContent=`AI: ${kept.length} kept…`;
    await new Promise(r=>setTimeout(r,10));
    if(typeof CfarDebug !== 'undefined') CfarDebug.updateCounts(cfarDets.length, yoloDets ? yoloDets.length : null, kept.length, null);

    // Step 4: Render
    const result  = Scene.renderDets(kept, filtered, { yoloRejected, pipeline });
    const totalMs = +(performance.now()-t0).toFixed(1);
    Met.record({ ...result, ms:totalMs, cfarMs:tCFAR, yoloMs:tYOLO, aiMs:tAI, pipeline });

    G('ov-st').textContent = 'complete';
    G('ov-final').textContent = kept.length;
    G('ov-ms').textContent = totalMs+'ms';
    G('s-ms').textContent  = totalMs+'ms';

    const prec = result.tp+result.fp>0 ? result.tp/(result.tp+result.fp) : 0;
    const rec  = result.tp+result.fn>0 ? result.tp/(result.tp+result.fn) : 0;
    const f1   = prec+rec>0 ? 2*prec*rec/(prec+rec) : 0;
    const tag  = { cfar:'CFAR', hybrid:'HYBRID', yolo_backend:'YOLO-B' }[pipeline]||'CFAR';
    const ySrc = YOLOInference.isSARModel() ? 'SAR' : (YOLOInference.getMode()==='simulate' ? 'SIM' : 'COCO');

    log(`━━ [${tag}/${ySrc}] CFAR:${cfarDets.length}${yoloDets?'→YOLO:'+yoloDets.length:''}→AI:${kept.length} (${totalMs}ms)`, 'info');
    log(`   TP=${result.tp} FP=${result.fp} FN=${result.fn} | P=${(prec*100).toFixed(0)}% R=${(rec*100).toFixed(0)}% F1=${(f1*100).toFixed(0)}%`, 'hit');
    if(tYOLO>0) log(`   Timing — CFAR:${tCFAR}ms YOLO:${tYOLO}ms AI:${tAI}ms`, 'ai');
    if(filtered && filtered.length) log(`   AI filtered ${filtered.length} FPs`, 'ai');
    if(yoloRejected.length) log(`   YOLO rejected ${yoloRejected.length} CFAR candidates`, 'ai');

      // ── TASK 7: FAILSAFE FOR NO LABELS ──
      if (result.tpDets.length === 0 && result.fpDets.length === 0) {
        log("⚠ No explicit TP/FP found — using weak labeling based on conf.", 'ai');
        try {
          cfarDets.forEach(d => {
            const isTP = (d.conf >= 0.6);
            AIModel.recordSample(d, isTP, W, H);
          });
        } catch(e) { console.error("AI sample record err:", e); }
        console.log(`Weak labels: ${cfarDets.length} added`);
      } else {
        // ── TASK 3 & 4: CONNECT TO AI MODEL ──
        try {
          for (const d of result.tpDets) AIModel.recordSample(d, true, W, H);
          for (const d of result.fpDets) AIModel.recordSample(d, false, W, H);
          if(filtered && filtered.length){
            for(const d of filtered){
              if(d.aiProb < 0.3) AIModel.recordSample(d, false, W, H);
            }
          }
        } catch(e) { console.error("AI sample record err:", e); }
      }
  
      // TASK 5: DEBUG LOGS
      console.log("CFAR dets:", cfarDets.length);
      console.log("Final dets:", kept.length);
      try {
        console.log("Samples:", AIModel.getSampleCount?.() || 0);
      } catch(e){}
      console.log("TP:", result.tpDets.length, "FP:", result.fpDets.length);
  
      // Task 3: Increase training frequency (Mini-batch)
      try {
        if(AIModel.getSampleCount && AIModel.getSampleCount() >= 4){ 
          AIModel.trainEpochs(10); 
        }
      } catch(e) { console.error("AI train err:", e); }
      updateAIDisplay();
  } catch(err){
    console.error('[runDetection] FATAL:', err);
    log('❌ Detection error: '+err.message, 'miss');
    G('ov-st').textContent = 'error';
    if(typeof CfarDebug !== 'undefined') CfarDebug.error('FATAL: ' + err.message);
    // FAILSAFE: regenerate scene so next attempt has fresh data
    try { Scene.gen(); } catch(_){}
  }

  pf.style.width='100%';
  await new Promise(r=>setTimeout(r,300));
  pw.style.display='none';
  runBtn.disabled = false;
  updateRunBtn(pipeline);
}

// ── AI toggle ──────────────────────────────────────────────────────
function toggleAI(){
  try {
    const on = AIModel.toggle();
    updateAIDisplay();
    log(`AI filter: ${on?'✓ ENABLED':'✗ DISABLED'}`, 'ai');
    if(CFAR.isArmed()) runDetection();
  } catch (e) {
    console.error("AIModel toggle err:", e);
  }
}

// ── Learn step ─────────────────────────────────────────────────────
async function learnStep(){
  try {
    const btn = G('learn-btn');
    const n   = AIModel.getSampleCount();
    if(n===0){
      btn.textContent='⚠ Run detection first to collect data';
      btn.style.borderColor='var(--red)'; btn.style.color='var(--red)';
      setTimeout(()=>updateAIDisplay(), 2000); return;
    }
    if(n<4){
      btn.textContent=`⚠ Need ≥4 samples, have ${n}`;
      btn.style.borderColor='var(--amber)'; btn.style.color='var(--amber)';
      setTimeout(()=>updateAIDisplay(), 2000); return;
    }
    btn.disabled=true; btn.style.borderColor='var(--amber)'; btn.style.color='var(--amber)';
    const {pos,neg}=AIModel.getSampleBreakdown();
    const epochsBefore=AIModel.getEpochs();
    for(let i=1;i<=10;i++){
      btn.textContent=`⟳ Training epoch ${i}/10… [${pos}✓ ${neg}✗]`;
      await new Promise(r=>setTimeout(r,18));
      AIModel.trainEpochs(1);
    }
    const result={
      loss:AIModel.getCurve().loss.slice(-1)[0]||0,
      acc: AIModel.getCurve().acc.slice(-1)[0]||0,
      f1:  AIModel.getCurve().f1.slice(-1)[0]||0,
    };
    const epoch=AIModel.getEpochs(), lr=AIModel.getLRNow();
    const {w,b}=AIModel.getWeights();
    const acc=AIModel.getAccuracy();
    log(`━━━━━━ LEARN COMPLETE ━━━━━━`, 'ai');
    log(`🧠 Epochs ${epochsBefore+1}→${epoch} | ${pos}✓ ${neg}✗`, 'ai');
    log(`   Loss:${result.loss.toFixed(5)} Acc:${result.acc.toFixed(1)}% F1:${result.f1.toFixed(1)}%`, 'ai');
    G('ai-epochs').textContent=epoch;
    G('ai-acc').textContent=acc!=null?(acc*100).toFixed(1)+'%':'—';
    G('ai-mode-display').textContent=`Ep${epoch} · F1 ${result.f1.toFixed(0)}%`;
    const aw=G('ai-weights');
    if(aw){
      const feat=['normArea','aspectRatio','yoloConf','sqrtArea','elongation','centerDist','conf','contrast'];
      aw.innerHTML=feat.map((f,i)=>`<span style="color:var(--dim)">${f}:</span> <span style="color:var(--purple)">${w[i].toFixed(4)}</span>`).join('<br>')+
        `<br><span style="color:var(--dim)">bias:</span> <span style="color:var(--purple)">${b.toFixed(4)}</span>`+
        `<br><span style="color:var(--dim)">lr:</span> <span style="color:var(--amber)">${lr.toFixed(6)}</span>`+
        `<br><span style="color:var(--dim)">F1:</span> <span style="color:var(--green)">${result.f1.toFixed(1)}%</span>`;
    }
    drawLearningCurve();
    btn.textContent=`✓ Done! Ep${epoch} · Loss ${result.loss.toFixed(3)} · F1 ${result.f1.toFixed(0)}%`;
    btn.style.borderColor='var(--green)'; btn.style.color='var(--green)'; btn.disabled=false;
    setTimeout(()=>updateAIDisplay(), 2500);
  } catch (e) {
    console.error("AIModel learn error:", e);
    const btn = G('learn-btn');
    if(btn) { btn.textContent = "⚠ Learn Error"; btn.disabled = false; }
  }
}

// ── Scene helpers ──────────────────────────────────────────────────
function newScene(){ Scene.gen(); clearStats(); log('New SAR scene generated','info'); }
function clearDets(){ Scene.gen(); clearStats(); }
function setScenario(i){
  Scene.setScenario(i); Scene.gen(); clearStats();
  log('Scenario: '+['Open Ocean','Cluttered Coast','Dense Formation','Rough Sea','Stealth Ships'][i],'info');
}
function clearStats(){
  ['s-tp','s-fp','s-fn'].forEach(id=>G(id).textContent='0');
  G('s-ai').textContent='0';
  ['s-prec','s-rec','s-ms'].forEach(id=>G(id).textContent='—');
  G('ov-final').textContent='0'; G('ov-ms').textContent='—';
  G('ov-st').textContent='ready';
}
function log(msg, cls='e'){
  const lg=G('dlog'), ts=new Date().toTimeString().slice(0,8);
  const el=document.createElement('div'); el.className=cls; el.textContent=`[${ts}] ${msg}`;
  lg.insertBefore(el, lg.firstChild);
  if(lg.children.length>80) lg.removeChild(lg.lastChild);
}

// ═══════════════════════════════════════════════════════════════════
// REAL SAR TAB — unchanged functional logic
// ═══════════════════════════════════════════════════════════════════
const RealSAR = (function(){
  let loadedImage=null, rawDets=[], _colorMode=false;
  let _autoParams = null; // auto-computed params from image analysis

  function setStatus(msg, color='var(--dim)'){
    const el=G('rs-status'); el.textContent=msg; el.style.color=color; el.style.borderColor=color;
  }

  function detectColorMode(id){ const d=id.data,n=d.length/4;let s=0,c=0;for(let i=0;i<n;i+=8){s+=Math.abs(d[i*4]-d[i*4+2]);c++;}return (s/c)>18; }

  // ── AUTO PARAMETER ANALYSIS ──────────────────────────────────────
  // Analyse image pixels to automatically determine best CFAR params
  function analyzeImageParams(id, W, H, colorMode) {
    const d = id.data, N = W * H;
    // Sample pixels for stats
    const step = Math.max(1, Math.floor(N / 8000));
    const samples = [];
    let orangeSum = 0, orangeCount = 0;

    for (let i = 0; i < N; i += step) {
      const r = d[i*4], g = d[i*4+1], b = d[i*4+2];
      const gray = r*0.299 + g*0.587 + b*0.114;
      samples.push(gray);
      if (colorMode) {
        // Measure orange channel signal (R-B difference weighted)
        const rb = r - b;
        if (rb > 8) { orangeSum += rb; orangeCount++; }
      }
    }
    samples.sort((a, b) => a - b);

    const p10 = samples[Math.floor(samples.length * 0.10)] || 0;
    const p50 = samples[Math.floor(samples.length * 0.50)] || 30;
    const p90 = samples[Math.floor(samples.length * 0.90)] || 200;
    const p98 = samples[Math.floor(samples.length * 0.98)] || 230;
    const dynamicRange = p90 - p10;

    // --- Orange threshold: percentile of orange signal distribution ---
    let oThr = 20; // default
    if (colorMode && orangeCount > 100) {
      const orangeSignals = [];
      for (let i = 0; i < N; i += step) {
        const r = d[i*4], b2 = d[i*4+2];
        const rb = r - b2;
        if (rb > 0) orangeSignals.push(rb);
      }
      orangeSignals.sort((a,b) => a - b);
      // Set threshold at 75th percentile of orange signals — keeps only strong orange blobs
      const p75 = orangeSignals[Math.floor(orangeSignals.length * 0.75)] || 20;
      oThr = Math.max(8, Math.min(80, Math.round(p75 * 0.9)));
    }

    // --- Window radius: scale with image size, adapt to noise level ---
    // Larger image → larger window; noisy (low range) → larger window
    const baseWin = colorMode ? 12 : 16;
    const sizeScale = Math.sqrt((W * H) / (600 * 400));
    const noiseAdj = dynamicRange < 60 ? 2 : (dynamicRange > 150 ? -2 : 0);
    const winR = Math.max(8, Math.min(24, Math.round(baseWin * Math.min(sizeScale, 1.4) + noiseAdj)));

    // --- Alpha: based on signal-to-noise ratio estimate ---
    const snrEst = p90 / Math.max(p50, 1);
    const alphaMult = colorMode ? 0.18 : 1.3;
    const N2 = (2*winR+1)**2 - 7**2; // approx guard 3
    const pfa = colorMode ? 5 : 10;
    const alphaBase = N2 * (Math.pow(pfa * 1e-4, -1/N2) - 1) * alphaMult;
    const alphaCap = (p90 * 0.85) / Math.max(p50, 1);
    const alpha = Math.min(alphaBase, Math.max(1.1, alphaCap));

    // --- Min area: scale with image resolution ---
    const minArea = Math.max(4, Math.round(8 * sizeScale));

    console.log(`[AutoParam] colorMode=${colorMode} oThr=${oThr} winR=${winR} alpha=${alpha.toFixed(2)} minArea=${minArea} DR=${dynamicRange.toFixed(0)} SNR=${snrEst.toFixed(2)}`);
    return { oThr, winR, alpha, minArea };
  }

  // Apply auto params to UI sliders with a visual flash
  function applyAutoParams(params) {
    _autoParams = params;

    const setSlider = (id, valId, val) => {
      const el = G(id); if (!el) return;
      el.value = val;
      const vEl = G(valId); if (vEl) vEl.textContent = val;
      // Flash the label to indicate auto-set
      el.style.accentColor = '#00e5ff';
      setTimeout(() => { el.style.accentColor = ''; }, 1200);
    };

    setSlider('rs-oth',  'rs-oth-v',  params.oThr);
    setSlider('rs-mina', 'rs-mina-v', params.minArea);

    // Show auto-param badge
    const badge = G('rs-autoparam-badge');
    if (badge) {
      badge.textContent = `⚡ Auto: oThr=${params.oThr} winR=${params.winR} minArea=${params.minArea}`;
      badge.style.display = 'block';
    }
  }

  function buildSignal(id,W,H,cm){ const d=id.data,px=new Float32Array(W*H);if(cm){for(let i=0;i<W*H;i++){const r=d[i*4],g=d[i*4+1],b=d[i*4+2],rb=r-b,gp=Math.max(0,g-r)*0.6,ws=rb>10?Math.max(0,rb-gp+Math.max(0,r-200)*0.3):0;px[i]=ws;}}else{for(let i=0;i<W*H;i++){px[i]=d[i*4]*0.299+d[i*4+1]*0.587+d[i*4+2]*0.114;}}return px; }
  function applyGrayEnhance(id){ const d=id.data,W=id.width,H=id.height,vals=new Float32Array(d.length/4);for(let i=0;i<vals.length;i++){const l=d[i*4]*0.299+d[i*4+1]*0.587+d[i*4+2]*0.114;vals[i]=Math.log1p(l);}const tW=Math.ceil(W/8),tH=Math.ceil(H/8),out=new Float32Array(vals.length);for(let ty=0;ty<8;ty++){for(let tx=0;tx<8;tx++){const x0=tx*tW,y0=ty*tH,x1=Math.min(x0+tW,W),y1=Math.min(y0+tH,H),tile=[];for(let y=y0;y<y1;y++)for(let x=x0;x<x1;x++)tile.push(vals[y*W+x]);tile.sort((a,b)=>a-b);const lo=tile[Math.floor(tile.length*0.02)]||0,hi=tile[Math.floor(tile.length*0.98)]||1,rng=hi-lo||1;for(let y=y0;y<y1;y++)for(let x=x0;x<x1;x++)out[y*W+x]=Math.max(0,Math.min(1,(vals[y*W+x]-lo)/rng));}}for(let i=0;i<out.length;i++){const v=Math.round(out[i]*255);d[i*4]=v;d[i*4+1]=v;d[i*4+2]=v;} }
  function cfarDetect(canvas,winR,guardR,kR,alpha,minArea,iouTh,cm,oThr){ const W=canvas.width,H=canvas.height,ctx=canvas.getContext('2d'),id=ctx.getImageData(0,0,W,H),rawPx=buildSignal(id,W,H,cm),px=new Float32Array(W*H);for(let y=1;y<H-1;y++){for(let x=1;x<W-1;x++){let s=0;for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++)s+=rawPx[(y+dy)*W+(x+dx)];px[y*W+x]=s/9;}}for(let x=0;x<W;x++){px[x]=rawPx[x];px[(H-1)*W+x]=rawPx[(H-1)*W+x];}for(let y=0;y<H;y++){px[y*W]=rawPx[y*W];px[y*W+W-1]=rawPx[y*W+W-1];}const logPx=new Float32Array(W*H);if(cm){let mL=0;for(let i=0;i<W*H;i++){logPx[i]=Math.log1p(px[i]);if(logPx[i]>mL)mL=logPx[i];}const ls=mL||1;for(let i=0;i<W*H;i++)logPx[i]=logPx[i]/ls*255;}else{for(let i=0;i<W*H;i++)logPx[i]=px[i];}const hF=cm?oThr:0,T=new Float32Array(W*H).fill(9999),step=3;for(let y=winR;y<H-winR;y+=step){for(let x=winR;x<W-winR;x+=step){const s=[];for(let dy=-winR;dy<=winR;dy++)for(let dx=-winR;dx<=winR;dx++){if(Math.abs(dy)<=guardR&&Math.abs(dx)<=guardR)continue;s.push(logPx[(y+dy)*W+(x+dx)]);}s.sort((a,b)=>a-b);const noise=s[Math.min(Math.floor(s.length*kR),s.length-1)],t=Math.max(hF,alpha*noise);for(let sy=0;sy<step&&y+sy<H;sy++)for(let sx=0;sx<step&&x+sx<W;sx++)T[(y+sy)*W+(x+sx)]=t;}}const bin=new Uint8Array(W*H);for(let i=0;i<W*H;i++){const mL=logPx[i]>T[i],mC=cm?(px[i]>hF):true;bin[i]=(mL&&mC)?1:0;}const dil=new Uint8Array(W*H);for(let y=1;y<H-1;y++)for(let x=1;x<W-1;x++){if(bin[y*W+x]||bin[(y-1)*W+x]||bin[(y+1)*W+x]||bin[y*W+x-1]||bin[y*W+x+1]||bin[(y-1)*W+x-1]||bin[(y-1)*W+x+1]||bin[(y+1)*W+x-1]||bin[(y+1)*W+x+1])dil[y*W+x]=1;}const lab=new Int32Array(W*H),par=[0];let next=1;const find=x=>{while(par[x]!==x){par[x]=par[par[x]];x=par[x];}return x;};const union=(a,b)=>{a=find(a);b=find(b);if(a!==b)par[a]=b;};for(let y=0;y<H;y++){for(let x=0;x<W;x++){if(!dil[y*W+x])continue;const L=x>0&&dil[y*W+x-1]?lab[y*W+x-1]:0,U=y>0&&dil[(y-1)*W+x]?lab[(y-1)*W+x]:0;if(!L&&!U){lab[y*W+x]=next;par.push(next);par[next]=next;next++;}else if(L&&!U)lab[y*W+x]=L;else if(!L&&U)lab[y*W+x]=U;else{lab[y*W+x]=L;union(L,U);}}}for(let i=0;i<W*H;i++)if(lab[i])lab[i]=find(lab[i]);const B={};for(let i=0;i<W*H;i++){const l=lab[i];if(!l)continue;const x=i%W,y=Math.floor(i/W);if(!B[l])B[l]={x1:x,y1:y,x2:x,y2:y,n:0,sum:0,maxV:0};const b=B[l];if(x<b.x1)b.x1=x;if(x>b.x2)b.x2=x;if(y<b.y1)b.y1=y;if(y>b.y2)b.y2=y;b.n++;b.sum+=logPx[i];if(logPx[i]>b.maxV)b.maxV=logPx[i];}let dets=[];for(const l in B){const b=B[l];if(b.n<minArea)continue;const dw=b.x2-b.x1+1,dh=b.y2-b.y1+1,ar=Math.max(dw,dh)/Math.max(Math.min(dw,dh),1);if(ar>14)continue;if(ar<1.05&&b.n>300)continue;const ms=b.sum/b.n,pmr=ms>0?b.maxV/ms:1;if(!cm&&b.n>40&&pmr<1.10)continue;const cx=Math.round((b.x1+b.x2)/2),cy=Math.round((b.y1+b.y2)/2),cl=T[cy*W+cx]||1,scr=ms/cl,sc=Math.min(0.97,0.38+Math.log1p(scr)*0.18),pad=cm?2:3;dets.push({x1:Math.max(0,b.x1-pad),y1:Math.max(0,b.y1-pad),x2:Math.min(W,b.x2+pad),y2:Math.min(H,b.y2+pad),conf:sc,brightness:+ms.toFixed(1),peakSig:+b.maxV.toFixed(1),peakMeanRatio:+pmr.toFixed(2),scr:+scr.toFixed(2)});}dets.sort((a,b)=>b.conf-a.conf);const keep=[],dead=new Set();for(let i=0;i<dets.length;i++){if(dead.has(i))continue;keep.push(dets[i]);for(let j=i+1;j<dets.length;j++){if(dead.has(j))continue;const a=dets[i],b=dets[j],ix1=Math.max(a.x1,b.x1),iy1=Math.max(a.y1,b.y1),ix2=Math.min(a.x2,b.x2),iy2=Math.min(a.y2,b.y2),inter=Math.max(0,ix2-ix1)*Math.max(0,iy2-iy1),u=(a.x2-a.x1)*(a.y2-a.y1)+(b.x2-b.x1)*(b.y2-b.y1)-inter;if(u>0&&inter/u>iouTh)dead.add(j);}}return keep; }
  function drawDets(dets,aiKept){ const cv=G('rs-canvas'),ctx=cv.getContext('2d');if(loadedImage&&!loadedImage._demo){ctx.drawImage(loadedImage,0,0,cv.width,cv.height);if(!_colorMode){const id2=ctx.getImageData(0,0,cv.width,cv.height);applyGrayEnhance(id2);ctx.putImageData(id2,0,0);}}const aiSet=new Set();for(const ak of aiKept){for(let i=0;i<dets.length;i++){if(dets[i].x1===ak.x1&&dets[i].y1===ak.y1){aiSet.add(i);break;}}}dets.forEach((d,i)=>{const isAI=aiSet.has(i),color=isAI?'#00e676':'#ff3b3b';ctx.strokeStyle=color;ctx.lineWidth=isAI?2.5:1.5;ctx.setLineDash(isAI?[]:[4,3]);ctx.strokeRect(d.x1,d.y1,d.x2-d.x1,d.y2-d.y1);ctx.setLineDash([]);if(isAI){ctx.fillStyle='rgba(0,230,118,.10)';ctx.fillRect(d.x1,d.y1,d.x2-d.x1,d.y2-d.y1);}ctx.fillStyle=color;ctx.font='bold 9px Consolas';const label=isAI?`${(d.conf*100).toFixed(0)}%`:'FP',ly=d.y1>12?d.y1-3:d.y2+10;ctx.fillText(label,d.x1+1,ly);}); }
  function updateTable(cfar,ai){ const tbl=G('rs-table');if(!ai.length){tbl.innerHTML='<span style="color:var(--dim)">No detections.</span>';return;}tbl.innerHTML=ai.map((d,i)=>{return`<div style="color:var(--green);border-bottom:1px solid var(--border);padding:2px 0">\u2713 #${i+1} [${d.x1},${d.y1}\u2192${d.x2},${d.y2}] conf=${(d.conf*100).toFixed(0)}%</div>`;}).join(''); }

  function loadFile(file){
    if(!file)return;
    setStatus('\u27f3 Loading\u2026','var(--amber)');
    const isJSON=/\.json$/i.test(file.name);
    if(isJSON){
      const r=new FileReader();
      r.onload=e=>{
        try{
          let p=JSON.parse(e.target.result);
          if(!Array.isArray(p)){p=p.detections||p.results||p.predictions||p.boxes||Object.values(p)[0]||[];}
          if(!Array.isArray(p))throw new Error('No detections array');
          const dets=p.map((d,i)=>{let x1,y1,x2,y2;if(d.x1!=null){x1=+d.x1;y1=+d.y1;x2=+d.x2;y2=+d.y2;}else if(d.xmin!=null){x1=+d.xmin;y1=+d.ymin;x2=+d.xmax;y2=+d.ymax;}else if(Array.isArray(d.bbox)){[x1,y1]=d.bbox;x2=x1+d.bbox[2];y2=y1+d.bbox[3];}else return null;const conf=+(d.confidence??d.conf??d.score??0.75).toFixed(2);return{x1:Math.round(x1),y1:Math.round(y1),x2:Math.round(x2),y2:Math.round(y2),conf,brightness:d.brightness??128};}).filter(Boolean);
          if(!dets.length)throw new Error('No valid detections');
          rawDets=dets;
          const cv=G('rs-canvas'),W=cv.width||800,H=cv.height||500,aiTh=+G('rs-aith').value;
          let kept=[];try{kept=AIModel.filter(dets,aiTh,W,H).kept||dets;}catch(err){console.error(err);kept=dets;}
          G('rs-final').textContent=kept.length;drawDets(dets,kept);updateTable(dets,kept);
          G('rs-overlay').style.display='none';
          setStatus(`\u2705 JSON \u2014 ${dets.length} dets \u00b7 ${kept.length} passed AI`,'var(--green)');
        }catch(err){setStatus('\u274c JSON parse: '+err.message,'var(--red)');}
      };
      r.readAsText(file);return;
    }
    const reader=new FileReader();
    reader.onload=e=>{
      const img=new Image();
      img.onload=()=>{
        const cv=G('rs-canvas'),container=cv.parentElement,maxW=container.clientWidth||600,scale=Math.min(1,maxW/img.naturalWidth);
        cv.width=Math.round(img.naturalWidth*scale);cv.height=Math.round(img.naturalHeight*scale);
        const ctx=cv.getContext('2d');ctx.drawImage(img,0,0,cv.width,cv.height);
        const rawId=ctx.getImageData(0,0,cv.width,cv.height);
        const modeEl=G('rs-mode'),manualMode=modeEl?modeEl.value:'auto';
        _colorMode=manualMode==='auto'?detectColorMode(rawId):(manualMode==='color');
        if(!_colorMode){applyGrayEnhance(rawId);ctx.putImageData(rawId,0,0);}
        const params = analyzeImageParams(rawId, cv.width, cv.height, _colorMode);
        applyAutoParams(params);
        G('rs-overlay').style.display='none';loadedImage=img;loadedImage._colorMode=_colorMode;
        G('rs-run').disabled=false;
        setStatus('\u2705 '+ (_colorMode?'False-color':'Grayscale') +' SAR \u2014 params auto-set, click Run','var(--green)');
      };
      img.onerror=function(){setStatus('\u274c Failed to load','var(--red)');};
      img.src=e.target.result;
    };
    reader.readAsDataURL(file);
  }

  async function runDetection(){
    const cv=G('rs-canvas');
    if(!loadedImage){setStatus('\u26a0 Load a SAR image first','var(--amber)');return;}
    if(loadedImage._demo){setStatus('\u26a0 Demo already detected','var(--amber)');return;}
    setStatus('\u27f3 Running CFAR + AI\u2026','var(--amber)');
    await new Promise(function(r){setTimeout(r,10);});
    const modeEl=G('rs-mode'),manualMode=modeEl?modeEl.value:'auto';
    if(manualMode!=='auto') _colorMode=(manualMode==='color');
    const aiTh=+G('rs-aith').value;
    const minArea=+G('rs-mina').value;
    const oThr=+(G('rs-oth')?G('rs-oth').value:20);
    const W=cv.width,H=cv.height;
    const ap = _autoParams;
    const winR  = ap ? ap.winR  : (_colorMode?12:16);
    const guardR = _colorMode ? 2 : 3;
    const alpha  = ap ? ap.alpha : (_colorMode ? 1.0 : 1.35);
    const t0=performance.now();
    const dets=cfarDetect(cv,winR,guardR,0.75,alpha,minArea,0.25,_colorMode,oThr);
    const ms=(performance.now()-t0).toFixed(0);
    rawDets=dets;
    var kept=[],filtered=[];
    try{var r=AIModel.filter(dets,aiTh,W,H);kept=r.kept||dets;filtered=r.filtered||[];}
    catch(e){console.error('AI error',e);kept=dets;}
    G('rs-final').textContent=kept.length;drawDets(dets,kept);updateTable(dets,kept);
    setStatus('\u2705 '+kept.length+' ships detected \u00b7 '+ms+'ms','var(--green)');
  }

  function loadDemo(){
    var cv=G('rs-canvas'),W=860,H=500;cv.width=W;cv.height=H;
    var ctx=cv.getContext('2d');_colorMode=true;
    var imgData=ctx.createImageData(W,H),d=imgData.data;
    for(var i=0;i<W*H;i++){var u1=Math.random()||1e-9,u2=Math.random();var spk=Math.sqrt(-2*Math.log(u1))*Math.cos(2*Math.PI*u2)*6+12;spk=Math.max(0,spk);var r=Math.round(spk*0.6+5),g=Math.round(spk*0.9+18+Math.random()*4),b=Math.round(spk*2.2+55+Math.random()*8);var x=i%W,coastX=x<230,coastLine=x+(Math.floor(i/W))<360;if(coastX&&coastLine){var in2=Math.random()*60+30;r=Math.round(r+in2*0.1);g=Math.round(g+in2*0.7);b=Math.round(b+in2*0.6);}d[i*4]=Math.min(255,r);d[i*4+1]=Math.min(255,g);d[i*4+2]=Math.min(255,b);d[i*4+3]=255;}
    ctx.putImageData(imgData,0,0);
    var ships=[{x:420,y:180,rw:9,rh:5,a:0.2},{x:455,y:172,rw:8,rh:4,a:-0.1},{x:490,y:190,rw:10,rh:5,a:0.35},{x:440,y:210,rw:7,rh:4,a:0.05},{x:475,y:205,rw:9,rh:5,a:-0.25},{x:510,y:175,rw:8,rh:4,a:0.15},{x:530,y:200,rw:11,rh:5,a:0.45},{x:460,y:235,rw:7,rh:3,a:-0.3},{x:500,y:225,rw:9,rh:4,a:0.1},{x:545,y:220,rw:8,rh:4,a:0.25}];
    ships.forEach(function(s){ctx.save();ctx.translate(s.x,s.y);ctx.rotate(s.a);var gr=ctx.createRadialGradient(0,0,0,0,0,Math.max(s.rw,s.rh));gr.addColorStop(0,'rgba(255,255,200,1.0)');gr.addColorStop(0.35,'rgba(255,200,60,0.95)');gr.addColorStop(0.70,'rgba(240,140,20,0.75)');gr.addColorStop(1.0,'rgba(200,80,0,0.05)');ctx.fillStyle=gr;ctx.beginPath();ctx.ellipse(0,0,s.rw,s.rh,0,0,Math.PI*2);ctx.fill();ctx.restore();});
    G('rs-overlay').style.display='none';
    loadedImage={_demo:true,_colorMode:true};
    rawDets=ships.map(function(s){return{x1:Math.max(0,Math.round(s.x-s.rw*Math.abs(Math.cos(s.a))-s.rh*Math.abs(Math.sin(s.a))-2)),y1:Math.max(0,Math.round(s.y-s.rh*Math.abs(Math.cos(s.a))-s.rw*Math.abs(Math.sin(s.a))-2)),x2:Math.min(W,Math.round(s.x+s.rw*Math.abs(Math.cos(s.a))+s.rh*Math.abs(Math.sin(s.a))+2)),y2:Math.min(H,Math.round(s.y+s.rh*Math.abs(Math.cos(s.a))+s.rw*Math.abs(Math.sin(s.a))+2)),conf:+(0.62+Math.random()*0.34).toFixed(2),brightness:Math.round(80+Math.random()*60)};});
    var aiTh=+G('rs-aith').value;
    var kept=[];try{kept=AIModel.filter(rawDets,aiTh,W,H).kept||rawDets;}catch(err){console.error(err);kept=rawDets;}
    G('rs-final').textContent=kept.length;
    drawDets(rawDets,kept);updateTable(rawDets,kept);
    G('rs-run').disabled=true;
    setStatus('\u2705 Demo SAR \u2014 '+kept.length+' ships detected','var(--green)');
  }
  return { loadFile: loadFile, runDetection: runDetection, loadDemo: loadDemo };
})();

function handleRSFile(file){ RealSAR.loadFile(file); }
function rsRunDetection(){ RealSAR.runDetection(); }
function rsLoadDemo(){ RealSAR.loadDemo(); }

// ── Init ───────────────────────────────────────────────────────────
window.addEventListener('load', () => {
  // 1. Initialize scene canvas (renders background + ships immediately)
  Scene.init(G('sc'));
  log('Scene ready · auto-arming detector…', 'info');
  log('YOLO: real-model only — checking backend…', 'ai');

  // 2. Update AI display and default pipeline
  updateAIDisplay();
  const psel = G('det-pipeline');
  if(psel) psel.value = 'hybrid';
  onPipelineChange('hybrid');   // default to hybrid pipeline

  // 3. Resize canvas when window resizes
  window.addEventListener('resize', () => Scene.gen());

  // 4. Verify scene canvas rendered correctly + AUTO-ARM
  setTimeout(() => {
    const cvs = Scene.getCanvas();
    if(cvs && cvs.width > 0 && cvs.height > 0){
      log(`Scene: ${cvs.width}×${cvs.height} ready for CFAR`, 'info');
      if(typeof CfarDebug !== 'undefined') CfarDebug.log(`Canvas ${cvs.width}×${cvs.height} — OK`);
    } else {
      log('⚠ Scene canvas size is 0 — regenerating…', 'miss');
      if(typeof CfarDebug !== 'undefined') CfarDebug.error('Canvas 0×0 — regenerating!');
      // FAILSAFE: force regenerate scene
      Scene.gen();
    }

    // AUTO-ARM: arm detector automatically so user can click Run immediately
    if(!CFAR.isArmed()){
      armDetector();
    }
  }, 300);

  // 5. Auto-init YOLO (pings backend, tries ONNX if offline)
  initYOLOPipeline();
});

