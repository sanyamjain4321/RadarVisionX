// ═══════════════════════════════════════════════════════════════════
// TRAINED AI MODEL — Production-grade Logistic Regression
// Architecture:
//   • 7 rich features per detection
//   • Adam optimizer (adaptive moment estimation)
//   • Binary Cross-Entropy loss with class-weight balancing
//   • Online feature normalization (running mean/std per feature)
//   • Stratified replay buffer (balanced TP/FP slots, max 400 samples)
//   • Mini-batch training (batch_size=16, shuffled each epoch)
//   • Learning rate schedule: cosine decay over epochs
//   • Auto-trains 5 epochs silently after every detection run
// ═══════════════════════════════════════════════════════════════════
const AIModel = (function(){
  const N_FEAT = 8; // Task 3 Feature expansion

  let w = [2.10, 0.55, 1.10, 1.20, 0.40, -0.65, -0.30, 0.85];
  let b = -1.10;

  const adam = {
    m_w: new Array(N_FEAT).fill(0),
    v_w: new Array(N_FEAT).fill(0),
    m_b: 0, v_b: 0,
    t: 0,
    beta1: 0.9, beta2: 0.999, eps: 1e-8
  };

  const norm = {
    mean: new Array(N_FEAT).fill(0),
    M2:   new Array(N_FEAT).fill(1),
    n: 0
  };

  const bufPos = [];
  const bufNeg = [];
  const BUF_MAX = 200;

  let epochs = 0;
  let enabled = true;
  let lrBase = 0.01;
  const BATCH = 16;

  const curve = { loss:[], acc:[], f1:[] };

  function sigmoid(z){ return 1/(1+Math.exp(-Math.max(-40,Math.min(40,z)))); }

  function bce(pred, label){
    const p = Math.max(1e-7, Math.min(1-1e-7, pred));
    return -(label*Math.log(p) + (1-label)*Math.log(1-p));
  }

  function updateNorm(raw){
    norm.n++;
    for(let i=0;i<N_FEAT;i++){
      const delta = raw[i] - norm.mean[i];
      norm.mean[i] += delta / norm.n;
      norm.M2[i]   += delta * (raw[i] - norm.mean[i]);
    }
  }

  function normalizeFeatures(raw){
    if(norm.n < 2) return [...raw];
    return raw.map((v,i)=>{
      const std = Math.sqrt(norm.M2[i]/Math.max(norm.n-1,1)) || 1;
      return (v - norm.mean[i]) / std;
    });
  }

  function extractRaw(det, W, H){
    const dw = det.x2 - det.x1, dh = det.y2 - det.y1;
    const area = dw * dh;
    const normArea = area / (W * H) * 1000;
    const ar = dw>0&&dh>0 ? dw/dh : 1.0; 
    const safeConf = (det.conf !== undefined && det.conf !== null) ? det.conf : 0.5;
    const sqrtArea = Math.sqrt(normArea);
    const elongation = dw>0&&dh>0 ? Math.max(dw,dh)/Math.max(Math.min(dw,dh),1) : 1;
    const cx=(det.x1+det.x2)/2, cy=(det.y1+det.y2)/2;
    const centerDist = Math.sqrt(((cx/W)-0.5)**2 + ((cy/H)-0.5)**2) / 0.707;
    
    // Task 3: Better Features & Robust extraction
    const intensityContrast = det.brightness !== undefined ? det.brightness / 255.0 : 0.5;
    const yoloConf = (det.yoloConf !== undefined && det.yoloConf !== null) ? det.yoloConf : safeConf;
    
    return [normArea, ar, yoloConf, sqrtArea, elongation, centerDist, safeConf, intensityContrast];
  }

  function extractFeatures(det, W, H){
    const raw = extractRaw(det, W, H);
    return normalizeFeatures(raw);
  }

  function predict(features){
    let z = b;
    for(let i=0;i<N_FEAT;i++) z += features[i]*w[i];
    return sigmoid(z);
  }

  function adamStep(grad_w, grad_b, lr){
    adam.t++;
    const bc1 = 1 - adam.beta1**adam.t;
    const bc2 = 1 - adam.beta2**adam.t;
    for(let i=0;i<N_FEAT;i++){
      adam.m_w[i] = adam.beta1*adam.m_w[i] + (1-adam.beta1)*grad_w[i];
      adam.v_w[i] = adam.beta2*adam.v_w[i] + (1-adam.beta2)*grad_w[i]**2;
      const m_hat = adam.m_w[i]/bc1;
      const v_hat = adam.v_w[i]/bc2;
      w[i] -= lr * m_hat / (Math.sqrt(v_hat) + adam.eps);
    }
    adam.m_b = adam.beta1*adam.m_b + (1-adam.beta1)*grad_b;
    adam.v_b = adam.beta2*adam.v_b + (1-adam.beta2)*grad_b**2;
    const mb_hat = adam.m_b/(1-adam.beta1**adam.t);
    const vb_hat = adam.v_b/(1-adam.beta2**adam.t);
    b -= lr * mb_hat / (Math.sqrt(vb_hat) + adam.eps);
  }

  function getLR(){
    const progress = Math.min(epochs / 200, 1);
    return lrBase * (0.05 + 0.95 * 0.5*(1 + Math.cos(Math.PI*progress)));
  }

  function buildBatch(batchSize){
    const half = Math.floor(batchSize/2);
    const pos = bufPos.length ? sampleN(bufPos, Math.min(half,bufPos.length)) : [];
    const neg = bufNeg.length ? sampleN(bufNeg, Math.min(batchSize-pos.length, bufNeg.length)) : [];
    return shuffle([...pos, ...neg]);
  }

  function sampleN(arr, n){
    const copy=[...arr]; shuffle(copy);
    return copy.slice(0,n);
  }
  function shuffle(arr){
    for(let i=arr.length-1;i>0;i--){const j=0|Math.random()*(i+1);[arr[i],arr[j]]=[arr[j],arr[i]];}
    return arr;
  }

  function classWeight(label){
    const nPos = bufPos.length || 1, nNeg = bufNeg.length || 1;
    const total = nPos + nNeg;
    return label===1 ? total/(2*nPos) : total/(2*nNeg);
  }

  function trainEpochs(nEpochs){
    const total = bufPos.length + bufNeg.length;
    if(total < 4) return null;

    let lastLoss = 0;
    for(let ep=0;ep<nEpochs;ep++){
      const batch = buildBatch(BATCH);
      if(!batch.length) break;

      const grad_w = new Array(N_FEAT).fill(0);
      let grad_b = 0, batchLoss = 0;

      for(const {features, label} of batch){
        const pred = predict(features);
        const cw = classWeight(label);
        const g = (pred - label) * cw;
        for(let i=0;i<N_FEAT;i++) grad_w[i] += g * features[i];
        grad_b += g;
        batchLoss += bce(pred, label) * cw;
      }

      const bs = batch.length;
      for(let i=0;i<N_FEAT;i++) grad_w[i] /= bs;
      grad_b /= bs;

      adamStep(grad_w, grad_b, getLR());
      lastLoss = batchLoss / bs;
      epochs++;

      const all = [...bufPos, ...bufNeg];
      let tp=0,fp=0,fn=0,tn=0;
      for(const {features,label} of all){
        const p = predict(features) >= 0.5 ? 1 : 0;
        if(p===1&&label===1) tp++;
        else if(p===1&&label===0) fp++;
        else if(p===0&&label===1) fn++;
        else tn++;
      }
      const acc = (tp+tn)/Math.max(all.length,1);
      const prec = tp/(tp+fp||1), rec = tp/(tp+fn||1);
      const f1 = 2*prec*rec/(prec+rec||1);
      curve.loss.push(+lastLoss.toFixed(5));
      curve.acc.push(+(acc*100).toFixed(1));
      curve.f1.push(+(f1*100).toFixed(1));
      if(curve.loss.length>100){ curve.loss.shift(); curve.acc.shift(); curve.f1.shift(); }
    }

    const all = [...bufPos, ...bufNeg];
    let tp=0,fp=0,fn=0,tn=0;
    for(const {features,label} of all){
      const p = predict(features) >= 0.5 ? 1 : 0;
      if(p===1&&label===1) tp++;
      else if(p===1&&label===0) fp++;
      else if(p===0&&label===1) fn++;
      else tn++;
    }
    const acc = (tp+tn)/Math.max(all.length,1);
    const prec = tp/(tp+fp||1), rec = tp/(tp+fn||1);
    const f1 = 2*prec*rec/(prec+rec||1);
    return { loss: lastLoss, acc, f1, tp, fp, fn, tn };
  }

  function trainEpoch(){ return trainEpochs(1); }

  function recordSample(det, isTP, W, H){
    const raw = extractRaw(det, W, H);
    updateNorm(raw);
    const features = normalizeFeatures(raw);
    const sample = {features, label: isTP ? 1 : 0};
    if(isTP){
      bufPos.push(sample);
      if(bufPos.length > BUF_MAX) bufPos.shift();
    } else {
      bufNeg.push(sample);
      if(bufNeg.length > BUF_MAX) bufNeg.shift();
    }
  }

  function getSampleCount(){ return bufPos.length + bufNeg.length; }
  function getSampleBreakdown(){ return {pos: bufPos.length, neg: bufNeg.length}; }
  function getCurve(){ return curve; }

  function filter(dets, baseThreshold, W, H){
    const kept=[], filtered=[];
    
    // Task 3: Dynamic Threshold based on recent FP rate tracking
    let dynamicThreshold = baseThreshold;
    if (bufNeg.length >= 20 && bufPos.length >= 5) {
        const acc = getAccuracy() || 0.5;
        // If accuracy is poor, tighten the threshold to filter out FPs
        // If accuracy is great, slightly relax to boost recall
        if (acc < 0.80) dynamicThreshold += 0.15;
        else if (acc > 0.92) dynamicThreshold -= 0.05;
        
        dynamicThreshold = Math.max(0.15, Math.min(0.95, dynamicThreshold));
    }
    
    for(const d of dets){
      const feat = extractFeatures(d, W, H);
      const prob = predict(feat);
      d.aiProb = prob;
      d.aiFeatures = feat;
      // Safeguard: Never drop if confidence is very high (>= 0.6)
      if(!enabled || prob >= dynamicThreshold || d.conf >= 0.6){ kept.push(d); }
      else { filtered.push(d); }
    }
    return {kept, filtered};
  }

  function getAccuracy(){
    const all=[...bufPos,...bufNeg];
    if(all.length<2) return null;
    let correct=0;
    for(const {features,label} of all){
      if((predict(features)>=0.5)===(label===1)) correct++;
    }
    return correct/all.length;
  }

  function getF1(){
    const all=[...bufPos,...bufNeg];
    if(all.length<2) return null;
    let tp=0,fp=0,fn=0;
    for(const {features,label} of all){
      const p=predict(features)>=0.5?1:0;
      if(p===1&&label===1)tp++;
      else if(p===1&&label===0)fp++;
      else if(p===0&&label===1)fn++;
    }
    const prec=tp/(tp+fp||1), rec=tp/(tp+fn||1);
    return 2*prec*rec/(prec+rec||1);
  }

  function getStats(){
    const all=[...bufPos,...bufNeg];
    let tp=0,fp=0,fn=0;
    for(const {features,label} of all){
      const p=predict(features)>=0.5?1:0;
      if(p===1&&label===1)tp++;
      else if(p===1&&label===0)fp++;
      else if(p===0&&label===1)fn++;
    }
    const prec=tp/(tp+fp||1), rec=tp/(tp+fn||1);
    const f1=2*prec*rec/(prec+rec||1);
    const acc = getAccuracy() || 0;
    return {
      accuracy: acc,
      precision: prec,
      recall: rec,
      f1: f1,
      epochs: epochs,
      samples: all.length
    };
  }

  function getWeights(){ return {w:[...w], b}; }
  function toggle(){ enabled=!enabled; return enabled; }
  function isEnabled(){ return enabled; }
  function getEpochs(){ return epochs; }
  function getLRNow(){ return getLR(); }

  return {
    filter, recordSample, trainEpoch, trainEpochs,
    predict, extractFeatures, extractRaw,
    getWeights, toggle, isEnabled, getEpochs, getStats,
    getAccuracy, getF1, getSampleCount, getSampleBreakdown, getCurve, getLRNow
  };
})();

