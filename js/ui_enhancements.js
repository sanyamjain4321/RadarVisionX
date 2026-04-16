/* ═══════════════════════════════════════════
   UI ENHANCEMENTS — Toast, Popup, Micro-FX
   ═══════════════════════════════════════════ */

// ── Toast system ────────────────────────────
const Toast = (() => {
  let container;
  const icons = { success:'✅', error:'❌', info:'📡', warn:'⚠️' };

  function ensure() {
    if (!container) {
      container = document.createElement('div');
      container.id = 'toast-container';
      document.body.appendChild(container);
    }
  }

  function show(type, title, msg, duration = 3800) {
    ensure();
    const t = document.createElement('div');
    t.className = `toast ${type}`;
    t.innerHTML = `
      <div class="toast-icon">${icons[type] || '💬'}</div>
      <div class="toast-body">
        <div class="toast-title">${title}</div>
        ${msg ? `<div class="toast-msg">${msg}</div>` : ''}
      </div>`;
    t.onclick = () => dismiss(t);
    container.appendChild(t);
    setTimeout(() => dismiss(t), duration);
    return t;
  }

  function dismiss(t) {
    if (t.classList.contains('hiding')) return;
    t.classList.add('hiding');
    setTimeout(() => t.remove(), 320);
  }

  return {
    success: (title, msg, d) => show('success', title, msg, d),
    error:   (title, msg, d) => show('error',   title, msg, d),
    info:    (title, msg, d) => show('info',     title, msg, d),
    warn:    (title, msg, d) => show('warn',     title, msg, d),
  };
})();

// ── Popup / Modal ───────────────────────────
const Popup = (() => {
  function show({ title, body, confirmText = 'OK', cancelText, onConfirm, onCancel } = {}) {
    const overlay = document.createElement('div');
    overlay.className = 'popup-overlay';

    overlay.innerHTML = `
      <div class="popup-box">
        <div class="popup-title">${title || 'Notice'}</div>
        <div class="popup-body">${body || ''}</div>
        <div class="popup-actions">
          ${cancelText ? `<button class="popup-btn secondary" id="popup-cancel">${cancelText}</button>` : ''}
          <button class="popup-btn primary" id="popup-confirm">${confirmText}</button>
        </div>
      </div>`;

    document.body.appendChild(overlay);

    const close = () => {
      overlay.style.animation = 'overlayIn 0.2s ease reverse';
      setTimeout(() => overlay.remove(), 200);
    };

    overlay.querySelector('#popup-confirm').onclick = () => {
      close(); onConfirm && onConfirm();
    };
    const cc = overlay.querySelector('#popup-cancel');
    if (cc) cc.onclick = () => { close(); onCancel && onCancel(); };
    overlay.onclick = e => { if (e.target === overlay) close(); };
  }
  return { show };
})();

// ── Ripple effect on buttons ─────────────────
document.addEventListener('click', e => {
  const btn = e.target.closest('.btn, #arm-btn, .popup-btn');
  if (!btn) return;
  const r = document.createElement('span');
  const rect = btn.getBoundingClientRect();
  const size = Math.max(rect.width, rect.height);
  r.style.cssText = `
    position:absolute;border-radius:50%;pointer-events:none;
    width:${size}px;height:${size}px;
    left:${e.clientX - rect.left - size/2}px;
    top:${e.clientY - rect.top - size/2}px;
    background:rgba(255,255,255,0.35);
    transform:scale(0);animation:rippleAnim 0.55s ease-out forwards;
    z-index:9;`;
  if (getComputedStyle(btn).position === 'static') btn.style.position = 'relative';
  btn.appendChild(r);
  r.addEventListener('animationend', () => r.remove());
});

// Inject ripple keyframes once
const rippleStyle = document.createElement('style');
rippleStyle.textContent = `@keyframes rippleAnim{to{transform:scale(2.8);opacity:0}}`;
document.head.appendChild(rippleStyle);

// ── Intercept arm button to show toast ───────
document.addEventListener('DOMContentLoaded', () => {
  // Wrap armDetector to show toast feedback
  const origArm = window.armDetector;
  if (origArm) {
    window.armDetector = function() {
      Toast.info('Arming Detector', 'Loading pipeline components…');
      origArm.apply(this, arguments);
    };
  }

  // Wrap runDetection to show result toast
  const origRun = window.runDetection;
  if (origRun) {
    window.runDetection = function() {
      origRun.apply(this, arguments);
    };
  }

  // Tab switch animation
  document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
      const panels = document.querySelectorAll('#tab-sar, #tab-realsar, #tab-metrics');
      panels.forEach(p => p.classList.remove('animate-in'));
      requestAnimationFrame(() => {
        panels.forEach(p => { if (p.style.display !== 'none') p.classList.add('animate-in'); });
      });
    });
  });
});

// ── Sidebar stat card counter animation ──────
function animateCount(el, to) {
  const from = parseFloat(el.textContent) || 0;
  const dur = 600; const start = performance.now();
  const isFloat = String(to).includes('.');
  function step(now) {
    const t = Math.min((now - start) / dur, 1);
    const ease = 1 - Math.pow(1 - t, 3);
    el.textContent = isFloat
      ? (from + (to - from) * ease).toFixed(2)
      : Math.round(from + (to - from) * ease);
    if (t < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}
window._animateCount = animateCount;

// Make Toast and Popup globally accessible
window.Toast = Toast;
window.Popup = Popup;
