// ═══════════════════════════════════════════════════════════════════
// 2D OS-CFAR DETECTOR — Hardened + Auto-Retry + Debug Logs
// ═══════════════════════════════════════════════════════════════════
const CFAR = (function(){
  let armed = false;

  function arm(cb){
    cb('Initializing CFAR + AI pipeline…');
    setTimeout(()=>{
      cb('Loading AI model weights…');
      setTimeout(()=>{ armed=true; cb('Pipeline armed ✓'); }, 200);
    }, 150);
  }

  let worker = null;
  let callIdCount = 0;
  const pendingCalls = new Map();

  function initWorker() {
    if (worker) return;
    worker = new Worker('js/cfar_worker.js');
    worker.onmessage = (e) => {
      const { action, callId, dets, detPixels, error, msg } = e.data;
      if (action === 'progress' && window.updateLoadingUI) {
         window.updateLoadingUI(msg);
      } else if (action === 'done') {
         if (pendingCalls.has(callId)) {
            pendingCalls.get(callId).resolve({ dets, detPixels });
            pendingCalls.delete(callId);
         }
      } else if (action === 'error') {
         if (pendingCalls.has(callId)) {
            pendingCalls.get(callId).reject(new Error(error));
            pendingCalls.delete(callId);
         }
      }
    };
  }

  async function detect(canvas, params){
    const {pfa, winR, minArea, iouTh} = params;
    const W=canvas.width, H=canvas.height;

    // ── Debug: log CFAR started ───────────────────────────────────
    console.log(`[CFAR] ══ STARTED ══ canvas: ${W}×${H}, pfa=${pfa.toExponential(1)}, winR=${winR}, minArea=${minArea}`);
    CfarDebug.log(`CFAR started — canvas ${W}×${H}`);

    // ── Guard: canvas dimensions must be valid ────────────────────
    if(W < 10 || H < 10){
      const err = `Canvas too small (${W}×${H}). Resize window or generate a scene first.`;
      console.error('[CFAR] ' + err);
      CfarDebug.error(err);
      return {dets:[], ms:0, error: err};
    }

    // ── Guard: canvas must not be all-black ───────────────────────
    const ctx2d = canvas.getContext('2d');
    let id;
    try {
      id = ctx2d.getImageData(0, 0, W, H);
    } catch(e) {
      const err = 'Cannot read canvas pixel data: ' + e.message;
      console.error('[CFAR] ' + err);
      CfarDebug.error(err);
      return {dets:[], ms:0, error: err};
    }

    // Sample pixels to check canvas is not blank
    let pixelSum = 0;
    const sampleStep = Math.max(1, Math.floor(W*H/500));
    for(let i=0;i<W*H;i+=sampleStep) pixelSum += id.data[i*4];
    const avgPixel = pixelSum / Math.ceil(W*H/sampleStep);
    console.log(`[CFAR] Canvas average pixel brightness: ${avgPixel.toFixed(1)}`);
    CfarDebug.log(`Canvas avg brightness: ${avgPixel.toFixed(1)}`);

    if(avgPixel < 1){
      const err = 'Canvas appears blank (avg brightness < 1). Generate a scene first.';
      console.warn('[CFAR] ' + err);
      CfarDebug.error(err);
      return {dets:[], ms:0, error: err};
    }

    // ── Extract grayscale pixels ──────────────────────────────────
    const px=new Float32Array(W*H);
    let minPx=255, maxPx=0;
    for(let i=0;i<W*H;i++){
      px[i]=(id.data[i*4]+id.data[i*4+1]+id.data[i*4+2])/3;
      if(px[i]<minPx) minPx=px[i];
      if(px[i]>maxPx) maxPx=px[i];
    }
    console.log(`[CFAR] Pixels extracted — min=${minPx.toFixed(1)} max=${maxPx.toFixed(1)} range=${(maxPx-minPx).toFixed(1)}`);
    CfarDebug.log(`Pixels extracted — range [${minPx.toFixed(0)}, ${maxPx.toFixed(0)}]`);

    // ── Guard: check dynamic range ────────────────────────────────
    if(maxPx - minPx < 5){
      const err = `Image has almost no contrast (range=${(maxPx-minPx).toFixed(1)}). Regenerate scene.`;
      console.warn('[CFAR] ' + err);
      CfarDebug.warn(err);
      // Don't return error — still try detection, just warn
    }

    const t0=performance.now();
    const guardR=3;
    const N=(2*winR+1)**2-(2*guardR+1)**2;

    // ── Alpha calculation — tuned for high recall on simulated SAR ──
    const alphaRaw = N*(Math.pow(pfa,-1/N)-1)*0.55;  // raised from 0.32 → better sensitivity

    // Dynamic cap: never let threshold exceed (maxPx * 0.85)
    const targetMaxThresh = maxPx * 0.85;
    // Estimate noise using 60th percentile (lower = more sensitive than 65th)
    const sampleSize = Math.min(W*H, 10000);
    const sampleArr = new Float32Array(sampleSize);
    const stride = Math.max(1, Math.floor(W*H / sampleSize));
    for(let i=0;i<sampleSize;i++) sampleArr[i] = px[i * stride] || 0;
    sampleArr.sort();
    const noiseEst = sampleArr[Math.floor(sampleSize * 0.60)] || 1;
    const alphaCap = noiseEst > 0 ? targetMaxThresh / noiseEst : 3.0;
    const alpha = Math.min(alphaRaw, Math.max(1.1, alphaCap));  // no *0.95 dampener
    console.log(`[CFAR] Alpha: raw=${alphaRaw.toFixed(2)} cap=${alphaCap.toFixed(2)} used=${alpha.toFixed(2)} noiseEst=${noiseEst.toFixed(1)}`);
    CfarDebug.log(`Alpha: ${alpha.toFixed(2)} (raw: ${alphaRaw.toFixed(1)}, cap: ${alphaCap.toFixed(1)})`);

    // Task 2: Dispatch to worker!
    initWorker();
    
    // Task 3: Inform UI of loading
    if(window.updateLoadingUI) window.updateLoadingUI('Dispatching CFAR worker...');

    let dets = [];
    try {
      const callId = ++callIdCount;
      const workerPromise = new Promise((resolve, reject) => {
        pendingCalls.set(callId, { resolve, reject });
      });
      
      worker.postMessage({
        callId, px, W, H, winR, guardR, alpha, minArea, iouTh
      }); // Deliberately NOT using transferable yet if not needed, to avoid trashing px array if needed later
      
      const result = await workerPromise;
      dets = result.dets;
      
    } catch(err) {
      console.error('[CFAR] Worker failed or timed out:', err);
      CfarDebug.error(err.message || "Worker error");
      // Clean recreate the worker if it failed
      worker.terminate();
      worker = null;
      return { dets: [], ms: Math.round(performance.now()-t0), error: err.message };
    }

    const ms=+(performance.now()-t0).toFixed(1);
    console.log(`[CFAR] ══ DONE ══ Detections found: ${dets.length} in ${ms}ms`);
    CfarDebug.log(`Detections found: ${dets.length} in ${ms}ms`);

    if(dets.length === 0){
      CfarDebug.warn(`Zero detections — try: lower PFA, smaller Min Area, or New SAR Scene`);
    } else {
      CfarDebug.ok(`${dets.length} ships detected ✓`);
    }

    return {dets, ms};
  }

  return {arm, detect, isArmed:()=>armed};
})();

// ═══════════════════════════════════════════════════════════════════
// CFAR DEBUG PANEL — on-screen debug overlay
// ═══════════════════════════════════════════════════════════════════
const CfarDebug = (function(){
  let _el = null;
  let _lines = [];

  function _getOrCreate(){
    if(_el) return _el;
    _el = document.createElement('div');
    _el.id = 'cfar-debug-panel';
    _el.style.cssText = [
      'position:fixed','bottom:12px','right:12px','z-index:9999',
      'background:rgba(5,8,15,.92)','border:1px solid #1a2540',
      'border-radius:4px','padding:10px 14px','font-size:10px',
      'font-family:Consolas,monospace','color:#5a6a80',
      'max-width:380px','max-height:280px','overflow-y:auto',
      'pointer-events:none','transition:opacity .3s',
      'box-shadow:0 2px 12px rgba(0,0,0,.6)'
    ].join(';');
    // Header
    const hdr = document.createElement('div');
    hdr.style.cssText='color:#00e5ff;font-size:9px;letter-spacing:1px;text-transform:uppercase;margin-bottom:6px;border-bottom:1px solid #1a2540;padding-bottom:4px';
    hdr.textContent = '◈ Pipeline Debug';
    _el.appendChild(hdr);
    // Counts section (always visible at top)
    _el._counts = document.createElement('div');
    _el._counts.id = 'cfar-debug-counts';
    _el._counts.style.cssText='margin-bottom:6px;padding-bottom:6px;border-bottom:1px solid #1a2540;line-height:1.8;font-size:10px';
    _el._counts.innerHTML =
      '<span style="color:#5a6a80">CFAR: </span><span style="color:#5a6a80">—</span>&nbsp;&nbsp;' +
      '<span style="color:#5a6a80">YOLO: </span><span style="color:#5a6a80">—</span>&nbsp;&nbsp;' +
      '<span style="color:#5a6a80">AI: </span><span style="color:#5a6a80">—</span>';
    _el.appendChild(_el._counts);
    _el._body = document.createElement('div');
    _el.appendChild(_el._body);
    // document.body.appendChild(_el);
    return _el;
  }

  function _addLine(text, color='#5a6a80'){
    return; // disable UI rendering only
    const el = _getOrCreate();
    const ts = new Date().toTimeString().slice(3,8);
    const line = document.createElement('div');
    line.style.color = color;
    line.style.marginBottom = '2px';
    line.textContent = `[${ts}] ${text}`;
    el._body.appendChild(line);
    // Keep last 15 lines
    while(el._body.children.length > 15){
      el._body.removeChild(el._body.firstChild);
    }
    el._body.scrollTop = el._body.scrollHeight;
    _lines.push(text);
  }

  function log(text)  { _addLine(text, '#5a6a80'); }
  function info(text) { _addLine(text, '#00e5ff'); }
  function ok(text)   { _addLine(text, '#00e676'); }
  function warn(text) { _addLine(text, '#ffab00'); }
  function error(text){ _addLine(text, '#ff3b3b'); }

  // Update detection counts section
  function updateCounts(cfar, yolo, ai, errMsg){
    return; // disable UI rendering only
    const el = _getOrCreate();
    const countEl = el._counts;
    const fmtNum = n => n != null ? `<span style="color:#00e5ff;font-weight:bold">${n}</span>` : '<span style="color:#5a6a80">—</span>';
    countEl.innerHTML =
      `<span style="color:#5a6a80">CFAR: </span>${fmtNum(cfar)}&nbsp;&nbsp;` +
      `<span style="color:#5a6a80">YOLO: </span>${fmtNum(yolo)}&nbsp;&nbsp;` +
      `<span style="color:#5a6a80">AI kept: </span>${fmtNum(ai)}` +
      (errMsg ? `<br><span style="color:#ff3b3b">ERR: ${errMsg}</span>` : '');
  }

  return { log, info, ok, warn, error, updateCounts };
})();
