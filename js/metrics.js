// ═══════════════════════════════════════════════════════════════════
// METRICS — Extended with pipeline mode tracking (CFAR / HYBRID / YOLO)
// ═══════════════════════════════════════════════════════════════════
const Met=(function(){
  const h=[];
  let totalTP=0,totalFP=0,totalFN=0,totalTN=0;

  function record(r){
    h.push(r);
    totalTP+=r.tp; totalFP+=r.fp; totalFN+=r.fn; totalTN+=(r.tn||0);
    render();
  }
  function avg(k){return h.reduce((s,x)=>s+(x[k]||0),0)/h.length;}
  function render(){
    if(!h.length)return;
    G('m-p').textContent=(avg('prec')*100).toFixed(1)+'%';
    G('m-r').textContent=(avg('rec')*100).toFixed(1)+'%';
    const f1s=h.map(x=>x.prec+x.rec>0?2*x.prec*x.rec/(x.prec+x.rec):0);
    G('m-f').textContent=(f1s.reduce((a,b)=>a+b,0)/f1s.length*100).toFixed(1)+'%';
    G('m-acc').textContent=(avg('acc')*100).toFixed(1)+'%';
    G('m-ms').textContent=avg('ms').toFixed(0)+'ms';

    G('cm-tp').textContent=totalTP;
    G('cm-fp').textContent=totalFP;
    G('cm-fn').textContent=totalFN;
    G('cm-tn').textContent=totalTN;

    const {w,b}=AIModel.getWeights();
    const feat_names=['normArea','AR','logConf','sqrtArea','elong','centerDist','conf'];
    G('ai-weights').innerHTML=
      feat_names.map((f,i)=>
        `<span style="color:var(--dim)">${f}:</span> <span style="color:var(--purple)">${w[i].toFixed(4)}</span>`
      ).join('<br>')+
      `<br><span style="color:var(--dim)">bias:</span> <span style="color:var(--purple)">${b.toFixed(4)}</span>`+
      `<br><span style="color:var(--dim)">epochs:</span> <span style="color:var(--cyan)">${AIModel.getEpochs()}</span>`+
      `<br><span style="color:var(--dim)">samples:</span> <span style="color:var(--amber)">${(()=>{const s=AIModel.getSampleBreakdown();return s.pos+'✓ '+s.neg+'✗';})()}</span>`;

    // ── Per-pipeline comparison table ─────────────────────────────────
    const modeLabels={cfar:'CFAR Only',hybrid:'CFAR→YOLO→AI',yolo_backend:'YOLO Backend'};
    const modeColors={cfar:'var(--cyan)',hybrid:'var(--blue)',yolo_backend:'var(--green)'};
    const modeData={};
    for(const m of ['cfar','hybrid','yolo_backend']){
      const runs=h.filter(x=>(x.pipeline||'cfar')===m);
      if(!runs.length) continue;
      const aP=runs.reduce((s,x)=>s+(x.prec||0),0)/runs.length;
      const aR=runs.reduce((s,x)=>s+(x.rec||0),0)/runs.length;
      const aF1=runs.map(x=>x.prec+x.rec>0?2*x.prec*x.rec/(x.prec+x.rec):0).reduce((a,b)=>a+b,0)/runs.length;
      const aMs=runs.reduce((s,x)=>s+(x.ms||0),0)/runs.length;
      modeData[m]={aP,aR,aF1,aMs,n:runs.length};
    }
    const cmpEl=G('mode-compare-table');
    if(cmpEl&&Object.keys(modeData).length){
      const hdr=`<div style="display:grid;grid-template-columns:1.2fr 1fr 1fr 1fr 80px 40px;gap:4px;border-bottom:1px solid var(--border);padding-bottom:3px;margin-bottom:3px;font-size:9px;color:var(--dim);letter-spacing:.5px"><span>Pipeline</span><span>Prec</span><span>Rec</span><span>F1</span><span>Avg ms</span><span>N</span></div>`;
      const rows=Object.entries(modeData).map(([m,d])=>
        `<div style="display:grid;grid-template-columns:1.2fr 1fr 1fr 1fr 80px 40px;gap:4px;padding:2px 0;font-size:11px">`+
        `<span style="color:${modeColors[m]}">${modeLabels[m]}</span>`+
        `<span style="color:var(--green)">${(d.aP*100).toFixed(1)}%</span>`+
        `<span style="color:var(--cyan)">${(d.aR*100).toFixed(1)}%</span>`+
        `<span style="color:var(--amber)">${(d.aF1*100).toFixed(1)}%</span>`+
        `<span style="color:var(--red)">${d.aMs.toFixed(0)}ms</span>`+
        `<span style="color:var(--dim)">${d.n}</span></div>`
      ).join('');
      cmpEl.innerHTML=hdr+rows;
    }

    G('rhist').innerHTML=h.map((x,i)=>{
      const pl=x.pipeline||'cfar';
      const mc={cfar:'var(--cyan)',hybrid:'var(--blue)',yolo_backend:'var(--green)'}[pl]||'var(--dim)';
      const f1=x.prec+x.rec>0?2*x.prec*x.rec/(x.prec+x.rec):0;
      return `<span style="color:var(--dim)">R${i+1}:</span> <span style="color:${mc}">[${pl.toUpperCase()}]</span>`+
        ` P=<span style="color:#16a34a;font-weight:600">${(x.prec*100).toFixed(0)}%</span>`+
        ` R=<span style="color:#06b6d4;font-weight:600">${(x.rec*100).toFixed(0)}%</span>`+
        ` F1=<span style="color:#f59e0b;font-weight:600">${(f1*100).toFixed(0)}%</span>`+
        ` TP=${x.tp} FP=${x.fp} FN=${x.fn} <span style="color:#ef4444;font-weight:600">${x.ms}ms</span>`;
    }).reverse().join('<br>');

    chart('prc',[{d:h.map(x=>x.prec*100),c:'#16a34a'},{d:h.map(x=>x.rec*100),c:'#06b6d4'}],100);
    chart('msc',[{d:h.map(x=>x.ms),c:'#ef4444'}]);
    drawLearningCurve();
  }
  function chart(id,series,maxY){
    const cv=G(id);if(!cv)return;
    const c=cv.getContext('2d'),W=cv.clientWidth||860,H=cv.clientHeight||130;
    cv.width=W;cv.height=H;
    c.fillStyle='#ffffff';c.fillRect(0,0,W,H);
    if(!series[0].d.length)return;
    const n=series[0].d.length,yMax=maxY||Math.max(...series.flatMap(s=>s.d))*1.15||1;
    
    // Grid and Labels
    c.strokeStyle='rgba(0,0,0,0.08)';c.lineWidth=1;
    c.font='10px Inter,sans-serif';c.fillStyle='#334155';
    for(let i=0;i<=4;i++){
      const y=H-H*i/4;
      c.beginPath();c.moveTo(0,y);c.lineTo(W,y);c.stroke();
      if(maxY===100) c.fillText(i*25+'%', 5, y-2);
    }
    // Axis lines
    c.strokeStyle='rgba(0,0,0,0.2)';c.lineWidth=2;
    c.beginPath();c.moveTo(0,H);c.lineTo(W,H);c.moveTo(0,0);c.lineTo(0,H);c.stroke();

    series.forEach(s=>{
      if(s.d.length<1)return;
      c.strokeStyle=s.c;c.lineWidth=3;c.lineJoin='round';c.lineCap='round';
      
      // Draw direct point-to-point line (Tension: 0)
      c.beginPath();
      const points = s.d.map((v,i)=>({x:i/(n-1||1)*W, y:H-Number(v)/yMax*H}));
      if(points.length>0){
        c.moveTo(points[0].x, points[0].y);
        for(let i=1; i<points.length; i++){
          c.lineTo(points[i].x, points[i].y);
        }
      }
      c.stroke();

      // Points
      points.forEach(p=>{
        c.fillStyle='#ffffff';c.beginPath();c.arc(p.x,p.y,5,0,Math.PI*2);c.fill();
        c.strokeStyle=s.c;c.lineWidth=2;c.beginPath();c.arc(p.x,p.y,4,0,Math.PI*2);c.stroke();
      });
    });
  }
  return{record, redraw:render};
})();
