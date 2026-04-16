// ═══════════════════════════════════════════════════════════════════
// SAR SCENE
// ═══════════════════════════════════════════════════════════════════
const Scene=(function(){
  let cvs,ctx,W,H,ships=[],scIdx=0;
  let _lastCallArgs = null, _mouseX = -1, _mouseY = -1, _hoverBox = null;
  let _bgCacheCanvas = document.createElement('canvas');
  let _animId = null, _pulseTime = 0;
  const SC=[
    {name:'Open Ocean',      ns:.28,cnt:12,cl:.025,st:0},
    {name:'Cluttered Coast', ns:.50,cnt:10,cl:.18, st:0},
    {name:'Dense Formation', ns:.36,cnt:20,cl:.05, st:0},
    {name:'Rough Sea',       ns:.70,cnt:8, cl:.10, st:0},
    {name:'Stealth Ships',   ns:.36,cnt:10,cl:.03, st:.65},
  ];

  function init(el){
    cvs=el;ctx=cvs.getContext('2d');
    
    cvs.addEventListener('mousemove', e => {
      const rect = cvs.getBoundingClientRect();
      _mouseX = (e.clientX - rect.left) * (cvs.width / rect.width);
      _mouseY = (e.clientY - rect.top) * (cvs.height / rect.height);
      if (_lastCallArgs && _lastCallArgs.tpDets) {
         let hb = null;
         for (const d of _lastCallArgs.tpDets) {
             const pad = 8;
             if (_mouseX >= d.x1-pad && _mouseX <= d.x2+pad && _mouseY >= d.y1-pad && _mouseY <= d.y2+pad) {
                 hb = d; break;
             }
         }
         if (hb !== _hoverBox) {
             _hoverBox = hb;
             redrawLast();
         }
      }
    });

    cvs.addEventListener('mouseleave', () => {
      _hoverBox = null;
      if (_lastCallArgs) redrawLast();
    });

    resize();gen();
  }
  
  function redrawLast() {
    if (_lastCallArgs) renderDets(_lastCallArgs.cfarDets, _lastCallArgs.aiFiltered, _lastCallArgs.extras);
  }

  function resize(){const w=cvs.parentElement;W=cvs.width=w.clientWidth||800;H=cvs.height=w.clientHeight||500;}

  function gen(){
    if (_animId) { cancelAnimationFrame(_animId); _animId = null; }
    ships=[];resize();
    const s=SC[scIdx];
    
    _bgCacheCanvas.width = W; 
    _bgCacheCanvas.height = H;
    const bgCtx = _bgCacheCanvas.getContext('2d');
    
    const prevCtx = ctx;
    ctx = bgCtx;
    _bg(s);_ships(s);_drawShips(s);_grid();
    ctx = prevCtx;
    
    ctx.clearRect(0,0,W,H);
    ctx.drawImage(_bgCacheCanvas, 0, 0);
    
    G('ov-gt').textContent=ships.length;
    G('ov-sc').textContent=s.name;
    G('ov-final').textContent='0';G('ov-st').textContent='ready';
    _lastCallArgs = null;
  }

  function _bg(s){
    const img=ctx.createImageData(W,H),d=img.data;
    for(let i=0;i<W*H;i++){
      const y2=Math.floor(i/W);
      // Task 6: Darker radar noise background
      let v=-Math.log(Math.random()||1e-9)*(12+s.ns*35)+Math.random()*8;
      v+=Math.sin(y2*.12)*1.0;
      if(Math.random()<s.cl*.005) v+=40+Math.random()*50;
      v=Math.max(0,Math.min(255,v));
      const g=Math.round(v);
      d[i*4]=g;d[i*4+1]=g;d[i*4+2]=g;d[i*4+3]=255;
    }
    ctx.putImageData(img,0,0);
  }

  function _ships(s){
    const cols=6,rows=3,cw=W/cols,ch=H/rows,cells=[];
    for(let r=0;r<rows;r++) for(let c=0;c<cols;c++) cells.push({r,c});
    for(let i=cells.length-1;i>0;i--){const j=0|Math.random()*(i+1);[cells[i],cells[j]]=[cells[j],cells[i]];}
    const cnt=Math.min(s.cnt,cells.length);
    for(let i=0;i<cnt;i++){
      const {r,c}=cells[i],mg=18;
      const x=c*cw+mg+Math.random()*(cw-2*mg), y=r*ch+mg+Math.random()*(ch-2*mg);
      const ang=Math.random()*Math.PI, len=20+Math.random()*30, wid=7+Math.random()*10;
      const st=Math.random()<s.st;
      ships.push({x,y,ang,len,wid,st,x1:x-len/2-5,y1:y-wid/2-5,x2:x+len/2+5,y2:y+wid/2+5});
    }
  }

  function _drawShips(s){
    ships.forEach(sh=>{
      ctx.save();ctx.translate(sh.x,sh.y);ctx.rotate(sh.ang);
      const a=sh.st?.25+Math.random()*.20:.90+Math.random()*.10;
      const g=ctx.createLinearGradient(-sh.len/2,0,sh.len/2,0);
      g.addColorStop(0,'rgba(185,208,232,.04)');
      g.addColorStop(.38,`rgba(225,238,255,${a})`);
      g.addColorStop(.62,`rgba(255,255,255,${a})`);
      g.addColorStop(1,'rgba(175,198,222,.04)');
      ctx.fillStyle=g;
      ctx.beginPath();
      ctx.moveTo(-sh.len/2,0);ctx.lineTo(-sh.len*.3,-sh.wid/2);
      ctx.lineTo(sh.len*.42,-sh.wid/2);ctx.lineTo(sh.len/2,0);
      ctx.lineTo(sh.len*.42,sh.wid/2);ctx.lineTo(-sh.len*.3,sh.wid/2);
      ctx.closePath();ctx.fill();
      if(!sh.st){
        ctx.strokeStyle=`rgba(175,200,230,${a*.4})`;ctx.lineWidth=1;
        ctx.beginPath();
        ctx.moveTo(-sh.len/2-8,2.5);ctx.lineTo(-sh.len/2-44,10);
        ctx.moveTo(-sh.len/2-8,-2.5);ctx.lineTo(-sh.len/2-44,-10);
        ctx.stroke();
      }
      ctx.restore();
    });
  }

  function _grid(){
    ctx.strokeStyle='rgba(37,99,235,0.06)';ctx.lineWidth=.5;
    for(let x=0;x<W;x+=60){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,H);ctx.stroke();}
    for(let y=0;y<H;y+=60){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke();}
  }

  function iou(a,b){
    const ix1=Math.max(a.x1,b.x1),iy1=Math.max(a.y1,b.y1);
    const ix2=Math.min(a.x2,b.x2),iy2=Math.min(a.y2,b.y2);
    const inter=Math.max(0,ix2-ix1)*Math.max(0,iy2-iy1);
    const u=(a.x2-a.x1)*(a.y2-a.y1)+(b.x2-b.x1)*(b.y2-b.y1)-inter;
    return u>0?inter/u:0;
  }

  function recordAIFeedback(tpDets, fpDets, aiFilteredDets){
    try {
      for(const d of tpDets) AIModel.recordSample(d, true, W, H);
      for(const d of fpDets) AIModel.recordSample(d, false, W, H);
      if(aiFilteredDets){
        for(const d of aiFilteredDets){
          if(d.aiProb < 0.30) AIModel.recordSample(d, false, W, H);
        }
      }
    } catch (e) {
      console.error("AIModel error:", e);
    }
  }

  function renderDets(cfarDetsOrig, aiFilteredOrig, extras={}){
    const mode = document.getElementById('v-mode')?.value || 'clean';
    
    // Performance limit (Task 8: max rendering)
    let cfarRender = [...cfarDetsOrig].sort((a,b)=>b.conf-a.conf).slice(0, 100);
    let aiFiltered = aiFilteredOrig ? [...aiFilteredOrig].slice(0, 50) : null;
    const { yoloRejected=[], pipeline='cfar' } = extras;
    let yoloRej = [...yoloRejected].slice(0, 50);

    const iouTh = +G('iou').value;

    const matched=new Set(),tpRaw=[],fpRaw=[];
    cfarRender.forEach(d=>{
      let bi=0,bj=-1;
      ships.forEach((sh,i)=>{if(matched.has(i))return;const ov=iou(d,sh);if(ov>bi){bi=ov;bj=i;}});
      if(bi>=iouTh&&bj>=0){matched.add(bj);tpRaw.push(d);}else fpRaw.push(d);
    });
    const fn=ships.filter((_,i)=>!matched.has(i));
    
    _lastCallArgs = { cfarDets: cfarDetsOrig, aiFiltered: aiFilteredOrig, extras, tpDets: tpRaw, fpDets: fpRaw, fnDets: fn, aiRej: aiFiltered, yoRej: yoloRej, mode, pipeline };

    if(!document.getElementById('learn-btn')?.disabled) {
       recordAIFeedback(tpRaw,fpRaw,aiFiltered);
    }

    const tpN=tpRaw.length,fpN=fpRaw.length,fnN=fn.length;
    const tnN=Math.max(0, ships.length*3 - tpN - fpN - fnN);
    const prec=tpN+fpN>0?tpN/(tpN+fpN):0;
    const rec=tpN+fnN>0?tpN/(tpN+fnN):0;
    const acc=tpN+fpN+fnN+tnN>0?(tpN+tnN)/(tpN+fpN+fnN+tnN):0;
    G('s-tp').textContent=tpN;G('s-fp').textContent=fpN;G('s-fn').textContent=fnN;
    G('s-ai').textContent=aiFilteredOrig?aiFilteredOrig.length:0;
    G('s-prec').textContent=(prec*100).toFixed(0)+'%';
    G('s-rec').textContent=(rec*100).toFixed(0)+'%';
    G('ov-final').textContent=cfarDetsOrig.length;

    if (_animId) cancelAnimationFrame(_animId);
    _pulseTime = 0;
    _animLoop();
    
    return{tp:tpN,fp:fpN,fn:fnN,tn:tnN,prec,rec,acc, tpDets:tpRaw, fpDets:fpRaw, fnDets:fn};
  }

  function _animLoop() {
    _animId = requestAnimationFrame(_animLoop);
    _pulseTime += 0.05;
    
    ctx.clearRect(0,0,W,H);
    ctx.drawImage(_bgCacheCanvas, 0, 0);
    
    if (!_lastCallArgs) return;
    const { tpDets, fpDets, fnDets, aiRej, yoRej, mode, pipeline } = _lastCallArgs;
    
    const drawBoxes = (arr, color, dash=[], bw=1) => {
       if(!arr) return;
       arr.forEach(d => {
          ctx.setLineDash(dash);
          ctx.strokeStyle = color;
          ctx.lineWidth = bw;
          ctx.strokeRect(d.x1, d.y1, d.x2-d.x1, d.y2-d.y1);
       });
    };

    if (mode === 'debug') {
        // Bottom: CFAR FP (Softer Red)
        drawBoxes(fpDets, 'rgba(239, 68, 68, 0.6)', [], 1.5);
        // AI Rejects (Purple dashed)
        drawBoxes(aiRej, 'rgba(124, 58, 237, 0.6)', [4,4], 1.5);
        // Yolo Rejects (Orange dashed)
        drawBoxes(yoRej, 'rgba(245, 158, 11, 0.6)', [3,3], 2.0);
        ctx.setLineDash([]);
    }

    // Top Level 1: Missed Ground Truth (Yellow)
    fnDets.forEach(sh => {
        ctx.setLineDash([4,4]);
        ctx.strokeStyle = '#f59e0b'; // Amber for Missed
        ctx.lineWidth = 2;
        const w = sh.x2 - sh.x1, h = sh.y2 - sh.y1;
        ctx.strokeRect(sh.x1, sh.y1, w, h);
        ctx.setLineDash([]);
        
        ctx.fillStyle = '#f59e0b';
        ctx.font = '10px Consolas';
        ctx.fillText('Missed', sh.x1, sh.y1 - 4);
    });
    
    // Top Level 2: True Positives (Green Pulsing Animation)
    const glow = 6 + Math.abs(Math.sin(_pulseTime)) * 8;
    const alpha = 0.7 + Math.abs(Math.sin(_pulseTime)) * 0.3;
    
    tpDets.forEach(d => {
        const isHover = _hoverBox && _hoverBox === d;
        
        ctx.shadowColor = 'rgba(34, 197, 94, 0.5)';
        ctx.shadowBlur = isHover ? 12 : 0;
        
        ctx.strokeStyle = `rgba(0, 230, 118, ${isHover ? 1 : alpha})`;
        ctx.lineWidth = isHover ? 3 : 2;
        ctx.strokeRect(d.x1, d.y1, d.x2-d.x1, d.y2-d.y1);
        
        ctx.shadowBlur = 0; // reset for text
        
        // Label formatting: "Ship ✓ (92%)"
        ctx.fillStyle = `rgba(0, 230, 118, ${isHover ? 1 : alpha})`;
        ctx.font = isHover ? 'bold 11px Consolas' : '10px Consolas';
        const confPerc = Math.round(d.conf*100);
        const txt = `Ship ✓ (${confPerc}%)`;
        const metrics = ctx.measureText(txt);
        
        const ty = isHover ? d.y2 + 14 : d.y1 - 4;
        ctx.fillStyle = 'rgba(255, 255, 255, 0.95)';
        ctx.fillRect(d.x1 - 1, ty - 10, metrics.width + 4, 14);
        
        ctx.fillStyle = `rgba(0, 230, 118, ${isHover ? 1 : alpha})`;
        ctx.fillText(txt, d.x1, ty);
        
        if (isHover && pipeline !== 'cfar' && d.yoloConf) {
           ctx.fillStyle = 'var(--blue)';
           ctx.fillText(`YOLS: ${Math.round(d.yoloConf*100)}%`, d.x1, ty + 12);
        }
    });
  }

  return{init,gen,renderDets,redrawLast,setScenario:i=>{scIdx=+i;},getCanvas:()=>cvs,getShips:()=>ships,getWH:()=>({W,H})};
})();

