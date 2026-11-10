/*
 * Engine side of the overlay: receives video frames from the content script (postMessage
 * with a transferred RGBA buffer), runs the analyzer, and posts back a plain-data snapshot.
 *
 * Messages in  (from the parent page's content script):
 *   {type: 'frame', w, h, buf, t}   RGBA pixels of one video frame, t in seconds
 *   {type: 'select', x, y}          Alt+click in frame pixels
 *   {type: 'clear' | 'reset' | 'orientation'}
 * Messages out: {type: 'ready'} | {type: 'state', snap, w, h, ms, selected?} | {type: 'error', error}
 */
(function () {
  'use strict';
  const IN = 'snooker-in', OUT = 'snooker-out';
  let analyzer = null, frameW = 0, frameH = 0;

  function post(msg) { window.parent.postMessage({ channel: OUT, msg }, '*'); }

  function cvReady() {
    return new Promise(resolve => {
      // The module object is a thenable that never settles cleanly; wait for the runtime instead
      if (self.cv && self.cv.Mat) { delete self.cv.then; resolve(); return; }
      self.cv.onRuntimeInitialized = () => { delete self.cv.then; resolve(); };
    });
  }

  function state(extra = {}) {
    post(Object.assign({ type: 'state', snap: analyzer.snapshot(), w: frameW, h: frameH }, extra));
  }

  function handle(m) {
    switch (m.type) {
      case 'frame': {
        const t0 = performance.now();
        const rgba = new cv.Mat(m.h, m.w, cv.CV_8UC4);
        rgba.data.set(new Uint8Array(m.buf));
        const rgb = new cv.Mat();
        cv.cvtColor(rgba, rgb, cv.COLOR_RGBA2RGB);
        rgba.delete();
        try {
          analyzer.process(rgb, m.t);
        } finally {
          rgb.delete();
        }
        frameW = m.w; frameH = m.h;
        state({ ms: performance.now() - t0 });
        break;
      }
      case 'select': {
        const t = analyzer.selectTarget([m.x, m.y]);
        state({ selected: t ? t.colour : null });
        break;
      }
      case 'clear': analyzer.clearTarget(); state(); break;
      case 'reset': analyzer.reset(); state(); break;
      case 'orientation': analyzer.toggleOrientation(); state(); break;
    }
  }

  cvReady().then(() => {
    analyzer = new Snooker.SnookerAnalyzer();
    window.addEventListener('message', e => {
      if (e.source !== window.parent) return;
      const d = e.data;
      if (!d || d.channel !== IN || !d.msg) return;
      try {
        handle(d.msg);
      } catch (err) {
        post({ type: 'error', error: String(err && err.message || err) });
      }
    });
    post({ type: 'ready' });
  });
})();
