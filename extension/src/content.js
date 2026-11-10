/*
 * Runs in the stream page. Finds the playing <video>, sends its frames to the recognition
 * engine (a hidden sandboxed extension iframe), and draws the result on a transparent canvas
 * laid exactly over the video picture, following resizes and full-screen.
 *
 * Controls (Option on a Mac):
 *   Alt + click a ball   select the target ball
 *   Alt + R              clear the target
 *   Alt + B              show / hide the ball colour rings (off by default; for checking recognition)
 *   Alt + O              switch portrait (broadcast view) / landscape (overhead view)
 *   Alt + A              forget the learned table views and learn again
 *   Alt + H / toolbar    show / hide the overlay
 *   Alt + S              save the current frame + what was recognised, for reporting problems
 */
(() => {
  'use strict';
  if (window.__snookerOverlay) return;
  window.__snookerOverlay = true;

  const ANALYSIS_MAX_WIDTH = 1280;  // frames are scaled down to at most this width for analysis
  const SEND_INTERVAL_MS = 100;     // how often to send a frame while the video plays
  const PAUSED_INTERVAL_MS = 400;   // ... and while it's paused (still useful for studying a position)

  const EXT_ORIGIN = new URL(chrome.runtime.getURL('')).origin;
  const STATE_TEXT = {
    locking: ['正在学习台面…（等主视角出现）', '#ffa500'],
    other_view: ['非全台画面，暂不画线', '#ff6060'],
    moving: ['球在滚动，等停稳…', '#ffa500'],
    settled: ['球已停稳', '#3f3'],
  };
  const BALL_COLOURS = {
    red: '#ff3c3c', yellow: '#ffe600', green: '#00c85a', brown: '#a05a28', blue: '#3c8cff',
    pink: '#ff82c8', black: '#282828', white: '#ffffff', unknown: '#969696',
  };
  const COLOUR_NAMES = {
    red: '红球', yellow: '黄球', green: '绿球', brown: '棕球', blue: '蓝球', pink: '粉球', black: '黑球', white: '白球', unknown: '未知',
  };

  let enabled = true, showBalls = false;  // per-ball colour rings are a debugging aid (Alt+B)
  let ready = false, busy = false, busySince = 0, lastSend = 0, blockedUntil = 0;
  let video = null, lastVideoScan = 0;
  let result = null;          // last {snap, w, h, ms} from the engine
  let message = null;         // transient notice {text, until}
  let error = null;           // persistent problem shown in the HUD

  // ---- engine iframe ------------------------------------------------------------------
  const iframe = document.createElement('iframe');
  iframe.src = chrome.runtime.getURL('src/engine.html');
  iframe.style.cssText = 'display:none';
  window.addEventListener('message', e => {
    if (e.source !== iframe.contentWindow) return;
    const d = e.data;
    if (!d || d.channel !== 'snooker-out' || !d.msg) return;
    const m = d.msg;
    if (m.type === 'ready') { ready = true; return; }
    if (m.type === 'error') { busy = false; error = '识别出错：' + m.error; return; }
    if (m.type === 'state') {
      busy = false;
      result = m;
      if ('selected' in m) notice(m.selected ? `已选目标：${COLOUR_NAMES[m.selected] || m.selected}` : '这里没有可选的球（白球不能当目标）');
    }
  });
  document.documentElement.appendChild(iframe);

  function send(msg, transfer = []) {
    if (iframe.contentWindow) iframe.contentWindow.postMessage({ channel: 'snooker-in', msg }, '*', transfer);
  }

  function notice(text) { message = { text, until: performance.now() + 2500 }; }

  // ---- overlay canvas -----------------------------------------------------------------
  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'position:fixed;pointer-events:none;z-index:2147483647;left:0;top:0;width:0;height:0;';
  const ctx = canvas.getContext('2d');

  function attachCanvas() {
    // In full-screen only the full-screen element's subtree is shown, so the canvas must live there
    const host = document.fullscreenElement || document.documentElement;
    if (canvas.parentNode !== host) host.appendChild(canvas);
  }
  document.addEventListener('fullscreenchange', attachCanvas);
  attachCanvas();

  // ---- video ----------------------------------------------------------------------------
  function findVideo() {
    let best = null, bestArea = 0;
    for (const v of document.querySelectorAll('video')) {
      if (!v.videoWidth) continue;
      const r = v.getBoundingClientRect();
      const area = r.width * r.height;
      if (area > bestArea && r.width > 50 && r.height > 50) { best = v; bestArea = area; }
    }
    return best;
  }

  /** On-screen rectangle of the actual picture inside the <video> box (letterboxing, object-fit). */
  function pictureRect(v) {
    const r = v.getBoundingClientRect();
    const vw = v.videoWidth, vh = v.videoHeight;
    const fit = getComputedStyle(v).objectFit;
    if (fit === 'fill') return { x: r.left, y: r.top, w: r.width, h: r.height, crop: null };
    const s = fit === 'cover' ? Math.max(r.width / vw, r.height / vh) : Math.min(r.width / vw, r.height / vh);
    const w = vw * s, h = vh * s;
    return { x: r.left + (r.width - w) / 2, y: r.top + (r.height - h) / 2, w, h, clip: r };
  }

  let captureCanvas = null, captureCtx = null;
  function captureFrame(v) {
    const scale = Math.min(1, ANALYSIS_MAX_WIDTH / v.videoWidth);
    const w = Math.round(v.videoWidth * scale), h = Math.round(v.videoHeight * scale);
    if (!captureCanvas || captureCanvas.width !== w || captureCanvas.height !== h) {
      captureCanvas = new OffscreenCanvas(w, h);
      captureCtx = captureCanvas.getContext('2d', { willReadFrequently: true });
    }
    captureCtx.drawImage(v, 0, 0, w, h);
    const img = captureCtx.getImageData(0, 0, w, h);   // throws if the video is protected
    return { w, h, buf: img.data.buffer };
  }

  // ---- input ----------------------------------------------------------------------------
  function insidePicture(e) {
    if (!video) return null;
    const p = pictureRect(video);
    if (e.clientX < p.x || e.clientX > p.x + p.w || e.clientY < p.y || e.clientY > p.y + p.h) return null;
    return p;
  }

  // Alt+click selects a ball; swallow the whole click so the player doesn't also pause/play
  for (const type of ['pointerdown', 'mousedown', 'pointerup', 'mouseup', 'click', 'dblclick']) {
    window.addEventListener(type, e => {
      if (!enabled || !e.altKey || !insidePicture(e)) return;
      e.preventDefault();
      e.stopImmediatePropagation();
      if (type === 'click' && result && result.w) {
        const p = pictureRect(video);
        send({ type: 'select', x: (e.clientX - p.x) / p.w * result.w, y: (e.clientY - p.y) / p.h * result.h });
      }
    }, true);
  }

  window.addEventListener('keydown', e => {
    if (!e.altKey || e.ctrlKey || e.metaKey) return;
    const actions = {
      KeyH: () => { enabled = !enabled; notice(enabled ? '画线已开启' : '画线已隐藏'); },
      KeyR: () => { send({ type: 'clear' }); notice('已取消目标球'); },
      KeyB: () => { showBalls = !showBalls; notice(showBalls ? '显示识别到的球' : '隐藏球圈'); },
      KeyO: () => { send({ type: 'orientation' }); notice('已切换台面朝向'); },
      KeyA: () => { send({ type: 'reset' }); notice('重新学习台面'); },
      KeyS: () => saveSnapshot(),
    };
    const act = actions[e.code];
    if (!act) return;
    if (e.code !== 'KeyH' && !enabled) return;
    e.preventDefault();
    e.stopImmediatePropagation();
    act();
  }, true);

  /**
   * Alt+S: downloads one JSON file with the current analysis-size frame (PNG data URL) and the
   * engine's latest result, so a wrong drawing can be reproduced and added to the tests.
   */
  function saveSnapshot() {
    if (!video) { notice('没有找到视频'); return; }
    let frameUrl = null;
    try {
      const scale = Math.min(1, ANALYSIS_MAX_WIDTH / video.videoWidth);
      const c = document.createElement('canvas');
      c.width = Math.round(video.videoWidth * scale);
      c.height = Math.round(video.videoHeight * scale);
      c.getContext('2d').drawImage(video, 0, 0, c.width, c.height);
      frameUrl = c.toDataURL('image/png');
    } catch (err) {
      notice('无法读取视频画面，没有保存'); return;
    }
    const stamp = new Date().toISOString().replace(/[-:]/g, '').replace('T', '_').slice(0, 15);
    const data = {
      savedAt: new Date().toISOString(),
      page: location.href,
      paused: video.paused,
      video: { width: video.videoWidth, height: video.videoHeight },
      result,
      frame: frameUrl,
    };
    const url = URL.createObjectURL(new Blob([JSON.stringify(data)], { type: 'application/json' }));
    const a = document.createElement('a');
    a.href = url;
    a.download = `snooker_snapshot_${stamp}.json`;
    document.documentElement.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
    notice(`已保存 snooker_snapshot_${stamp}.json 到下载文件夹`);
  }

  chrome.runtime.onMessage.addListener(m => {
    if (m && m.type === 'toggle') { enabled = !enabled; notice(enabled ? '画线已开启' : '画线已隐藏'); }
  });

  // ---- main loop --------------------------------------------------------------------------
  function tick() {
    requestAnimationFrame(tick);
    const now = performance.now();
    if (now - lastVideoScan > 1000 || (video && !video.isConnected)) {
      lastVideoScan = now;
      const v = findVideo();
      if (v !== video) { video = v; result = null; }
    }
    attachCanvas();

    if (busy && now - busySince > 5000) busy = false;  // engine hiccup: don't stall forever
    if (enabled && ready && !busy && video && video.readyState >= 2 && now >= blockedUntil) {
      const interval = video.paused ? PAUSED_INTERVAL_MS : SEND_INTERVAL_MS;
      if (now - lastSend >= interval) {
        lastSend = now;
        try {
          const f = captureFrame(video);
          busy = true; busySince = now;
          send({ type: 'frame', w: f.w, h: f.h, buf: f.buf, t: now / 1000 }, [f.buf]);
          if (error && error.startsWith('无法读取')) error = null;
        } catch (err) {
          error = '无法读取视频画面（该视频可能受版权保护）';
          blockedUntil = now + 5000;
        }
      }
    }
    draw();
  }

  // ---- drawing ----------------------------------------------------------------------------
  function draw() {
    if (!enabled || !video) { canvas.style.width = '0px'; canvas.style.height = '0px'; return; }
    const p = pictureRect(video);
    const dpr = window.devicePixelRatio || 1;
    const cw = Math.round(p.w * dpr), ch = Math.round(p.h * dpr);
    if (canvas.width !== cw || canvas.height !== ch) { canvas.width = cw; canvas.height = ch; }
    Object.assign(canvas.style, { left: p.x + 'px', top: p.y + 'px', width: p.w + 'px', height: p.h + 'px' });
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, cw, ch);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const lines = [];
    let headColour = '#fff';
    if (!ready) {
      lines.push(['正在加载识别引擎…', '#ffa500']);
    } else if (result) {
      const s = result.snap;
      const [text, colour] = STATE_TEXT[s.state] || [s.state, '#fff'];
      lines.push([text + (s.state === 'settled' || s.state === 'moving' ? `　${s.balls.length} 颗球` : ''), colour]);
      headColour = colour;
      drawShot(s, p.w / result.w, p.h / result.h);
      if (s.targetColour) lines.push([`目标：${COLOUR_NAMES[s.targetColour] || s.targetColour}`, '#ff0']);
      if (s.plan) {
        lines.push(s.plan.blocked
          ? [`被挡住：路线上有 ${s.plan.blockers.length} 颗球`, '#ff4040']
          : [`路线畅通，切角 ${Math.round(s.plan.cutDeg)}°`, '#3f3']);
      } else if (s.noPocket) {
        lines.push(['切角太薄，没有能进的袋', '#ffa500']);
      } else if (s.state === 'settled' && s.targetId === null) {
        lines.push(['按住 Alt（Mac 上是 Option）点一颗球作为目标', '#ddd']);
      }
    } else {
      lines.push(['等待视频画面…', '#ffa500']);
    }
    if (error) lines.push([error, '#ff6060']);
    if (message && performance.now() < message.until) lines.push([message.text, '#0ff']);
    drawHud(lines, headColour);
  }

  function drawShot(s, sx, sy) {
    const P = pt => [pt[0] * sx, pt[1] * sy];
    const rs = (sx + sy) / 2;
    const live = s.state === 'settled' || s.state === 'moving';
    if (live && showBalls) {
      for (const b of s.balls) {
        const [x, y] = P([b.x, b.y]);
        ctx.strokeStyle = BALL_COLOURS[b.colour] || BALL_COLOURS.unknown;
        ctx.lineWidth = 2;
        ctx.setLineDash(b.visible ? [] : [3, 3]);
        ctx.beginPath(); ctx.arc(x, y, b.r * rs + 4, 0, 2 * Math.PI); ctx.stroke();
      }
      ctx.setLineDash([]);
    }
    // Mark the cue ball and the target only once a target is chosen, to keep the picture clean
    if (live && s.targetId !== null) {
      for (const [id, colour] of [[s.cueId, '#fff'], [s.targetId, '#ff0']]) {
        const b = s.balls.find(t => t.id === id);
        if (!b) continue;
        const [x, y] = P([b.x, b.y]);
        ctx.strokeStyle = colour; ctx.lineWidth = 3;
        ctx.beginPath(); ctx.arc(x, y, b.r * rs + 7, 0, 2 * Math.PI); ctx.stroke();
      }
    }
    const plan = s.plan;
    if (!plan) return;
    // One-ball-wide corridors the balls sweep; red where another ball is in the way
    for (const [poly, blocked, colour] of [[plan.aimCorridor, plan.aimBlocked, 'rgba(255,255,255,0.22)'],
                                          [plan.potCorridor, plan.potBlocked, 'rgba(255,255,0,0.22)']]) {
      if (!poly) continue;
      ctx.fillStyle = blocked ? 'rgba(255,0,0,0.35)' : colour;
      ctx.beginPath();
      poly.forEach((q, i) => { const [x, y] = P(q); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
      ctx.closePath(); ctx.fill();
    }
    const line = (a, b, colour, dash) => {
      ctx.strokeStyle = colour; ctx.lineWidth = 2; ctx.setLineDash(dash);
      ctx.beginPath(); ctx.moveTo(...P(a)); ctx.lineTo(...P(b)); ctx.stroke();
    };
    line(plan.aim[0], plan.aim[1], 'rgba(255,255,255,0.9)', [8, 6]);      // aim line
    line(plan.pot[0], plan.pot[1], 'rgba(255,255,0,0.9)', []);            // pot line
    line(plan.cuePath[0], plan.cuePath[1], 'rgba(0,255,255,0.9)', [2, 5]); // cue ball after contact
    ctx.setLineDash([]);
    ctx.strokeStyle = '#f00'; ctx.lineWidth = 3;
    for (const b of plan.blockers) { const [x, y] = P(b); ctx.beginPath(); ctx.arc(x, y, 14, 0, 2 * Math.PI); ctx.stroke(); }
  }

  function drawHud(lines, headColour) {
    ctx.font = '13px -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif';
    const title = '斯诺克画线';
    const width = Math.max(ctx.measureText(title).width, ...lines.map(l => ctx.measureText(l[0]).width)) + 16;
    const height = 8 + 18 * (lines.length + 1);
    ctx.fillStyle = 'rgba(0,0,0,0.6)';
    ctx.fillRect(6, 6, width, height);
    ctx.fillStyle = headColour;
    ctx.fillText(title, 14, 24);
    lines.forEach(([text, colour], i) => { ctx.fillStyle = colour; ctx.fillText(text, 14, 24 + 18 * (i + 1)); });
  }

  requestAnimationFrame(tick);
})();
