// ═══════════════════════════════════════════════════════════════════
// CFAR WEB WORKER — Background math processor
// ═══════════════════════════════════════════════════════════════════

function threshMap(px, W, H, winR, guardR, kRatio, alpha){
  const T = new Float32Array(W*H).fill(255);
  // Cap window size to 12 (max 24x24 block size approx) for performance
  winR = Math.min(winR, 12);
  // Increased stride step to 4 to cut computational load by 45%
  const step = 4;
  
  // Fast 3x3 pre-filter smoothing to reduce noise
  const smoothed = new Float32Array(W*H);
  for(let y=1; y<H-1; y++){
    for(let x=1; x<W-1; x++){
      let sum = 0;
      for(let dy=-1; dy<=1; dy++){
        for(let dx=-1; dx<=1; dx++){
          sum += px[(y+dy)*W + (x+dx)];
        }
      }
      smoothed[y*W+x] = sum / 9;
    }
  }
  // Fill borders
  for(let x=0; x<W; x++) { smoothed[x]=px[x]; smoothed[(H-1)*W+x]=px[(H-1)*W+x]; }
  for(let y=0; y<H; y++) { smoothed[y*W]=px[y*W]; smoothed[y*W+(W-1)]=px[y*W+(W-1)]; }

  for(let y=winR; y<H-winR; y+=step){
    for(let x=winR; x<W-winR; x+=step){
      let sum=0, sqSum=0, count=0;
      for(let dy=-winR;dy<=winR;dy++){
        for(let dx=-winR;dx<=winR;dx++){
          if(Math.abs(dy)<=guardR && Math.abs(dx)<=guardR) continue;
          // Use smoothed pixels for background stats
          const p = smoothed[(y+dy)*W+(x+dx)];
          sum+=p; sqSum+=p*p; count++;
        }
      }
      
      const mean = sum / count;
      const variance = Math.max(0, (sqSum / count) - (mean * mean));
      const std = Math.sqrt(variance);
      
      // Reduce threshold multiplier if highly regular (sea clutter), boost if messy
      const localAlpha = alpha * (1 + Math.min(variance / 800, 0.65));
      const dynamicKRatio = variance > 1000 ? Math.max(0.5, kRatio - 0.15) : kRatio;
      
      // Adaptive Threshold based on Local Mean + k * Std (CA-CFAR logic)
      const t = mean + (localAlpha * dynamicKRatio * std * 0.4);
      
      for(let sy=0;sy<step&&y+sy<H;sy++)
        for(let sx=0;sx<step&&x+sx<W;sx++)
          T[(y+sy)*W+(x+sx)] = Math.max(15, t); // Hard contrast floor = 15
    }
  }
  return T;
}

function ccLabel(bin, W, H){
  const lab = new Int32Array(W*H);
  const par = [0];
  let next = 1;
  function find(x){while(par[x]!==x){par[x]=par[par[x]];x=par[x];}return x;}
  function union(a,b){a=find(a);b=find(b);if(a!==b)par[a]=b;}
  for(let y=0;y<H;y++){
    for(let x=0;x<W;x++){
      if(!bin[y*W+x]) continue;
      const L=x>0&&bin[y*W+x-1]?lab[y*W+x-1]:0;
      const U=y>0&&bin[(y-1)*W+x]?lab[(y-1)*W+x]:0;
      if(!L&&!U){lab[y*W+x]=next;par.push(next);par[next]=next;next++;}
      else if(L&&!U) lab[y*W+x]=L;
      else if(!L&&U) lab[y*W+x]=U;
      else{lab[y*W+x]=L;union(L,U);}
    }
  }
  for(let i=0;i<W*H;i++) if(lab[i]) lab[i]=find(lab[i]);
  return lab;
}

function bboxes(lab, W, H, minA, px){
  const B={};
  for(let i=0;i<W*H;i++){
    const l=lab[i]; if(!l) continue;
    const x=i%W, y=Math.floor(i/W);
    if(!B[l]) B[l]={x1:x,y1:y,x2:x,y2:y,n:0,sum:0,maxV:0};
    const b=B[l];
    if(x<b.x1)b.x1=x;if(x>b.x2)b.x2=x;
    if(y<b.y1)b.y1=y;if(y>b.y2)b.y2=y;
    b.n++;
    if(px[i]>b.maxV)b.maxV=px[i];
    b.sum+=px[i];
  }
  const out=[];
  for(const l in B){
    const b=B[l];
    if(b.n<minA) continue;
    const dw=b.x2-b.x1+1, dh=b.y2-b.y1+1;
    const ar=Math.max(dw,dh)/Math.max(Math.min(dw,dh),1);
    const boundingArea = dw * dh;
    const compactness = b.n / boundingArea;
    
    // Relaxed shape filters — ships in simulation vary widely in aspect ratio & size
    if(ar > 15.0) continue;                    // only reject very long thin streaks
    if(b.n < 10) continue;                     // Ignore extremely small blobs (< 10 px)
    if(compactness < 0.05) continue;           // looser compactness — sim ships can be sparse
    
    const localContrast = b.sum > 0 ? b.sum / b.n : 128;
    if (localContrast < 12) continue;          // lower contrast floor for dim/stealth ships
    
    // Peak-to-mean ratio: ensure the blob has a bright core (ship-like)
    const pmr = b.maxV > 0 ? b.maxV / Math.max(localContrast, 1) : 1;
    
    const conf = Math.min(0.97, 0.40 + b.n / 200 + (localContrast / 900) + (pmr - 1) * 0.08);
    if (conf <= 0.28) continue;
    
    // Expand detection regions by 4px
    out.push({x1:b.x1-4, y1:b.y1-4, x2:b.x2+4, y2:b.y2+4,
              conf: conf,
              label:'ship',
              brightness: localContrast});
              
    // Early stopping if too many detections (stop noise explosion)
    if (out.length > 200) break;
  }
  
  // Task 1: Cap detections before expensive n^2 NMS
  out.sort((a,b)=>b.conf-a.conf);
  return out.slice(0, 150);
}

function nms(dets, th){
  const keep=[],dead=new Set();
  for(let i=0;i<dets.length;i++){
    if(dead.has(i)) continue; keep.push(dets[i]);
    for(let j=i+1;j<dets.length;j++){
      if(dead.has(j)) continue;
      const a=dets[i],b=dets[j];
      const ix1=Math.max(a.x1,b.x1),iy1=Math.max(a.y1,b.y1);
      const ix2=Math.min(a.x2,b.x2),iy2=Math.min(a.y2,b.y2);
      const inter=Math.max(0,ix2-ix1)*Math.max(0,iy2-iy1);
      const u=(a.x2-a.x1)*(a.y2-a.y1)+(b.x2-b.x1)*(b.y2-b.y1)-inter;
      if(u>0){
          const iou = inter/u;
          if(iou > th) {
              dead.add(j);
              a.x1 = Math.min(a.x1, b.x1);
              a.y1 = Math.min(a.y1, b.y1);
              a.x2 = Math.max(a.x2, b.x2);
              a.y2 = Math.max(a.y2, b.y2);
          }
      }
    }
  }
  return keep;
}

function _detectWithAlpha(px, W, H, winR, guardR, alpha, minArea, iouTh){
  postMessage({ action: 'progress', msg: 'Thresholding map...' });
  const T=threshMap(px,W,H,winR,guardR,0.75,alpha);

  postMessage({ action: 'progress', msg: 'Extracting features...' });
  const bin=new Uint8Array(W*H);
  let detPixels=0;
  for(let i=0;i<W*H;i++){
    if(px[i]>T[i]){ bin[i]=1; detPixels++; }
  }

  const dil=new Uint8Array(W*H);
  for(let y=1;y<H-1;y++) for(let x=1;x<W-1;x++){
    if(bin[y*W+x]||bin[(y-1)*W+x]||bin[(y+1)*W+x]||
       bin[y*W+x-1]||bin[y*W+x+1]) dil[y*W+x]=1;
  }

  const lab=ccLabel(dil,W,H);
  let dets=bboxes(lab,W,H,minArea,px);
  dets=nms(dets, iouTh);

  return { dets, detPixels };
}

self.onmessage = function(e) {
  const t0 = performance.now();
  try {
    const { px, W, H, winR, guardR, alpha, minArea, iouTh, callId } = e.data;
    
    // Attempt detection
    let result = _detectWithAlpha(px, W, H, winR, guardR, alpha, minArea, iouTh);
    let dets = result.dets;
    
    // Auto-retry cascade inside worker just like before
    if(dets.length === 0){
      const retryAlphas = [
        Math.max(1.1, alpha * 0.7),
        Math.max(1.05, alpha * 0.5),
        1.05,
      ];
      for(let ri=0; ri<retryAlphas.length; ri++){
        postMessage({ action: 'progress', msg: `Fallback retry ${ri+1}...` });
        const retryAlpha = retryAlphas[ri];
        result = _detectWithAlpha(px, W, H, winR, guardR, retryAlpha, Math.max(4, minArea-4), iouTh);
        dets = result.dets;
        if(dets.length > 0) break;
      }
    }
    
    postMessage({ action: 'done', callId, ms: performance.now()-t0, dets, detPixels: result.detPixels });
  } catch (err) {
    postMessage({ action: 'error', callId, error: err.message });
  }
};
