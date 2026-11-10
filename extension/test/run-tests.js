/*
 * Tests for the extension's recognition code (src/*.js) in Node, on the same screenshots and
 * ground truth as the Python version's test_stream.py, so both implementations are held to
 * the same expectations.
 *
 * Usage (from the project folder):  node extension/test/run-tests.js
 */
'use strict';
const fs = require('fs');
const path = require('path');
const zlib = require('zlib');

const PROJECT = path.resolve(__dirname, '..', '..');
const EXT = path.resolve(__dirname, '..');

// ---- tiny PNG reader (8-bit RGB/RGBA, non-interlaced: what screenshots are) ------------

function readPNG(file) {
  const buf = fs.readFileSync(file);
  let pos = 8, width, height, bitDepth, colorType, interlace;
  const idat = [];
  while (pos < buf.length) {
    const len = buf.readUInt32BE(pos), type = buf.toString('ascii', pos + 4, pos + 8);
    const data = buf.subarray(pos + 8, pos + 8 + len);
    if (type === 'IHDR') {
      width = data.readUInt32BE(0); height = data.readUInt32BE(4);
      bitDepth = data[8]; colorType = data[9]; interlace = data[12];
    } else if (type === 'IDAT') idat.push(data);
    else if (type === 'IEND') break;
    pos += 12 + len;
  }
  if (bitDepth !== 8 || (colorType !== 2 && colorType !== 6) || interlace) throw new Error(`unsupported PNG: ${file}`);
  const bpp = colorType === 6 ? 4 : 3, stride = width * bpp;
  const raw = zlib.inflateSync(Buffer.concat(idat));
  const out = new Uint8Array(width * height * 3);
  const prev = new Uint8Array(stride), cur = new Uint8Array(stride);
  for (let y = 0; y < height; y++) {
    const filter = raw[y * (stride + 1)];
    const line = raw.subarray(y * (stride + 1) + 1, (y + 1) * (stride + 1));
    for (let i = 0; i < stride; i++) {
      const a = i >= bpp ? cur[i - bpp] : 0, b = prev[i], c = i >= bpp ? prev[i - bpp] : 0;
      let v = line[i];
      if (filter === 1) v += a;
      else if (filter === 2) v += b;
      else if (filter === 3) v += (a + b) >> 1;
      else if (filter === 4) { const p = a + b - c, pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c); v += pa <= pb && pa <= pc ? a : pb <= pc ? b : c; }
      cur[i] = v & 255;
    }
    for (let x = 0; x < width; x++) for (let k = 0; k < 3; k++) out[(y * width + x) * 3 + k] = cur[x * bpp + k];
    prev.set(cur);
  }
  return { width, height, data: out };
}

// ---- OpenCV.js + extension code ---------------------------------------------------------

function loadCV() {
  const cv = require(path.join(EXT, 'lib', 'opencv.js'));
  return new Promise(resolve => {
    // The module object is a thenable that never settles cleanly; wait for the runtime instead
    if (cv.Mat) { delete cv.then; resolve(cv); return; }
    cv.onRuntimeInitialized = () => { delete cv.then; resolve(cv); };
  });
}

const failures = [];
function check(name, ok, detail = '') {
  console.log(`  [${ok ? 'PASS' : 'FAIL'}] ${name}${detail ? `  (${detail})` : ''}`);
  if (!ok) failures.push(name);
}

(async () => {
  const cv = await loadCV();
  globalThis.cv = cv;
  for (const f of ['vision.js', 'geometry.js', 'tracking.js', 'analyzer.js']) require(path.join(EXT, 'src', f));
  const S = globalThis.Snooker;

  const imread = rel => { const p = readPNG(path.join(PROJECT, rel)); return cv.matFromArray(p.height, p.width, cv.CV_8UC3, p.data); };
  const resized = (m, w, h, interp = cv.INTER_LINEAR) => { const o = new cv.Mat(); cv.resize(m, o, new cv.Size(w, h), 0, 0, interp); return o; };
  const gt = JSON.parse(fs.readFileSync(path.join(PROJECT, 'test_data', 'ground_truth.json'), 'utf8')).images;

  // ---------------- simulated stream --------------------------------------------------
  function removeBall(img, x, y, r) {
    const mask = cv.Mat.zeros(img.rows, img.cols, cv.CV_8UC1);
    cv.circle(mask, new cv.Point(Math.trunc(x), Math.trunc(y + r * 0.3)), Math.trunc(r * 1.7), new cv.Scalar(255), -1);
    const out = new cv.Mat();
    cv.inpaint(img, mask, out, 7, cv.INPAINT_TELEA);
    mask.delete();
    return out;
  }
  function moveBall(img, x, y, r, nx, ny) {
    const k = Math.trunc(r * 1.3);
    const patch = img.roi(new cv.Rect(Math.trunc(x) - k, Math.trunc(y) - k, 2 * k, 2 * k)).clone();
    const out = removeBall(img, x, y, r);
    const disc = cv.Mat.zeros(2 * k, 2 * k, cv.CV_8UC1);
    cv.circle(disc, new cv.Point(k, k), Math.trunc(r * 1.05), new cv.Scalar(255), -1);
    const region = out.roi(new cv.Rect(Math.trunc(nx) - k, Math.trunc(ny) - k, 2 * k, 2 * k));
    patch.copyTo(region, disc);
    region.delete(); patch.delete(); disc.delete();
    return out;
  }

  function streamTest() {
    console.log('Simulated stream (main view: 梯形桌面截屏.png)');
    const main = imread('梯形桌面截屏.png');
    const h = main.rows, w = main.cols, S112 = 1.12;
    const blue = [1091 * S112, 458 * S112];
    const crop = main.roi(new cv.Rect(Math.trunc(w * 0.35), Math.trunc(h * 0.55), Math.trunc(w * 0.8) - Math.trunc(w * 0.35), h - Math.trunc(h * 0.55)));
    const closeup = resized(crop, w, h); crop.delete();
    const overhead0 = imread('斯诺克.png'), overhead = resized(overhead0, w, h); overhead0.delete();
    const page0 = imread('huya.png'), page = resized(page0, w, h); page0.delete();

    const a = new S.SnookerAnalyzer();
    let t = 0;
    const play = (frame, seconds) => { for (let i = 0; i < Math.trunc(seconds * 20); i++) { a.process(frame, t); t += 0.05; } return a; };

    play(main, 0.2);
    check('not drawing before the table is learned', a.state === 'locking', a.state);
    play(main, 1.5);
    check('table learned and balls settled', a.state === 'settled', a.state);
    check('19 balls tracked', a.tracker.balls().length === 19, String(a.tracker.balls().length));
    const cue = a.cueBall;
    check('cue ball found', !!cue && Math.hypot(cue.pos[0] - 1113 * S112, cue.pos[1] - 889 * S112) < 15);
    const sel = a.selectTarget([blue[0] - 10, blue[1] + 8]);
    check('click selects the blue', !!sel && sel.colour === 'blue', sel ? sel.colour : 'none');
    play(main, 0.1);
    check('blue shot is blocked', !!a.plan && a.plan.blocked, a.plan ? `${a.plan.blockers.length} blockers` : 'no plan');

    for (const [name, frame] of [['player close-up / part of table', closeup], ['overhead camera', overhead], ['non-table frame', page]]) {
      play(frame, 0.6);
      check(`cut to ${name}: lines hidden`, a.state === 'other_view' && !a.plan, a.state);
    }
    play(main, 0.1);
    check('cut back: main view recognised immediately', a.state === 'settled', a.state);
    check('cut back: blue still selected and plan shown', !!a.target && a.target.colour === 'blue' && !!a.plan);

    const covered = main.clone();
    cv.rectangle(covered, new cv.Point(Math.trunc(1060 * S112), Math.trunc(700 * S112)), new cv.Point(Math.trunc(1320 * S112), Math.trunc(830 * S112)), new cv.Scalar(30, 30, 30), -1);
    play(covered, 1.0);
    check('player covering reds: still settled', a.state === 'settled', a.state);
    check('covered reds remembered', a.tracker.balls().length === 19, String(a.tracker.balls().length));
    check('covered reds still block the shot', !!a.plan && a.plan.blocked);
    play(main, 0.5);

    const r = a.target.r;
    let moved = null;
    const states = [];
    for (let i = 1; i <= 5; i++) {
      const frame = moveBall(main, blue[0], blue[1], r, blue[0] + 40 * i, blue[1] - 25 * i);
      play(frame, 0.2);
      states.push(a.state);
      if (moved) moved.delete();
      moved = frame;
    }
    check('while the blue rolls: lines hidden', states.includes('moving') && !a.plan, states.join(','));
    check('blue moved away: selection cleared', a.target === null);
    play(moved, 1.0);
    check('balls settle again', a.state === 'settled', a.state);

    // The cue ball comes to rest touching a red, then rolls away: the marker must follow the white
    {
      const b2 = new S.SnookerAnalyzer();
      let t2 = 0;
      const play2 = (frame, seconds) => { for (let i = 0; i < Math.trunc(seconds * 20); i++) { b2.process(frame, t2); t2 += 0.05; } };
      const cue0 = [1113 * S112, 889 * S112], red = [1227 * S112, 972 * S112], rr = 18.5;
      const dv = [cue0[0] - red[0], cue0[1] - red[1]], dl = Math.hypot(...dv);
      const touching = [red[0] + dv[0] / dl * 2 * rr, red[1] + dv[1] / dl * 2 * rr];
      let f = moveBall(main, cue0[0], cue0[1], rr, touching[0], touching[1]);
      play2(f, 2.0); f.delete();
      let c = b2.cueBall;
      check('cue ball resting against a red: found', !!c && Math.hypot(c.pos[0] - touching[0], c.pos[1] - touching[1]) < 12,
            c ? `at ${c.pos.map(Math.round)}` : 'none');
      for (const i of [1, 2, 3]) {
        const q = [touching[0] + (cue0[0] - touching[0]) * i / 4, touching[1] + (cue0[1] - touching[1]) * i / 4];
        f = moveBall(main, cue0[0], cue0[1], rr, q[0], q[1]); play2(f, 0.2); f.delete();
      }
      play2(main, 1.5);
      c = b2.cueBall;
      check('cue ball rolls away from the red: marker follows the white', !!c && Math.hypot(c.pos[0] - cue0[0], c.pos[1] - cue0[1]) < 12,
            c ? `at ${c.pos.map(Math.round)}` : 'none');
      const nearRed = b2.tracker.balls().filter(t => Math.hypot(t.pos[0] - red[0], t.pos[1] - red[1]) < 12);
      check('the red it touched is still a red', nearRed.length === 1 && nearRed[0].colour === 'red', nearRed.map(t => t.colour).join(','));
    }

    const pink = [549 * S112, 898 * S112];
    const potted = removeBall(moved, pink[0], pink[1], r * 1.1);
    play(potted, 1.0);
    const colours = a.tracker.balls().map(b => b.colour);
    check('potted pink forgotten', !colours.includes('pink') && colours.length === 18, `${colours.length} balls`);
    for (const m of [main, closeup, overhead, page, covered, moved, potted]) m.delete();
  }

  // ---------------- robustness ----------------------------------------------------------
  function robustnessTest() {
    console.log('Robustness: detection + colour labels on degraded copies');
    const variants = {
      'original': m => m.clone(),
      'half size': m => resized(m, Math.round(m.cols * 0.5), Math.round(m.rows * 0.5), cv.INTER_AREA),
      'darker (x0.6)': m => { const o = new cv.Mat(); cv.convertScaleAbs(m, o, 0.6, 0); return o; },
      'brighter (x1.3)': m => { const o = new cv.Mat(); cv.convertScaleAbs(m, o, 1.3, 0); return o; },
      'blurry': m => { const o = new cv.Mat(); cv.GaussianBlur(m, o, new cv.Size(7, 7), 2.5); return o; },
    };
    for (const img of gt) {
      const base = imread(img.image);
      for (const [vname, fn] of Object.entries(variants)) {
        const im = fn(base), k = im.cols / base.cols;
        const v = new S.SnookerVision(), g = new S.SnookerGeometry();
        g.setOrientation(img.portrait);
        const { ok, quad } = v.detectMainView(im);
        if (!ok) { check(`${img.image} ${vname}: table found`, false); im.delete(); continue; }
        g.updateHomography(S.TableLock.sortCorners(quad));
        const found = v.findBalls(im, g.playingArea(), (x, y) => g.ballRadiusPx(x, y));
        let nFound = 0, right = 0;
        const wrong = [];
        for (const b of img.balls) {
          const x = b.x * k, y = b.y * k;
          let best = null;
          found.forEach((f, i) => { const d = Math.hypot(f[0] - x, f[1] - y); if (!best || d < best[0]) best = [d, i]; });
          if (best && best[0] < 12 * k + 4) {
            nFound++;
            if (found[best[1]][3] === b.colour) right++; else wrong.push(`${b.colour}->${found[best[1]][3]}`);
          }
        }
        const extra = found.length - nFound;
        check(`${img.image} ${vname}: ${nFound}/${img.balls.length} found, ${right}/${img.balls.length} colours right, ${extra} extra`,
              nFound === img.balls.length && right === img.balls.length && extra === 0, wrong.join(', '));
        im.delete();
      }
      base.delete();
    }
  }

  function palePinkTest() {
    console.log('Pale pink vs cue ball (test_data/直播实测_主视角_球员遮挡.png)');
    const base = imread('test_data/直播实测_主视角_球员遮挡.png');
    const pink = [519, 356], cue = [683, 549];
    for (const wash of [0.25, 0.35, 0.45, 0.55, 0.65, 0.75]) {
      const im = base.clone();
      const d = im.data, w = im.cols;
      for (let y = pink[1] - 8; y <= pink[1] + 8; y++) for (let x = pink[0] - 8; x <= pink[0] + 8; x++) {
        if ((x - pink[0]) ** 2 + (y - pink[1]) ** 2 > 64) continue;
        for (let c = 0; c < 3; c++) { const i = (y * w + x) * 3 + c; d[i] = Math.trunc(d[i] * (1 - wash) + 255 * wash); }
      }
      const v = new S.SnookerVision(), g = new S.SnookerGeometry();
      const { quad } = v.detectMainView(im);
      g.updateHomography(S.TableLock.sortCorners(quad));
      const found = v.findBalls(im, g.playingArea(), (x, y) => g.ballRadiusPx(x, y));
      const at = p => found.filter(b => Math.hypot(b[0] - p[0], b[1] - p[1]) < 8).map(b => b[3]);
      check(`pink washed ${Math.round(wash * 100)}% towards white: still pink, cue ball still white`,
            at(pink).join() === 'pink' && at(cue).join() === 'white', `pink -> ${at(pink)}, cue -> ${at(cue)}`);
      im.delete();
    }
    base.delete();
  }

  function pocketMouthTest() {
    console.log('Pocket mouth');
    const g = new S.SnookerGeometry();
    g.updateHomography([[120, 60], [880, 70], [960, 680], [50, 700]]);
    const centre = [g.targetW / 2, g.targetH / 2];
    g.pockets.forEach((jaw, i) => {
      const mouth = g.pocketMouth(i);
      const further = Math.hypot(mouth[0] - centre[0], mouth[1] - centre[1])
        > Math.hypot(jaw[0] - centre[0], jaw[1] - centre[1]) + 5;
      check(`pocket ${i} opening is past the jaw`, further);
    });
    const plan = g.planShot([400, 500], [700, 220], []);
    const mouth = g.pocketMouth(plan.pocketIndex);
    const cloth = g.toCamera(mouth);
    check('pot line ends at the opening on the cloth',
          Math.hypot(plan.pocket[0] - cloth[0], plan.pocket[1] - cloth[1]) < 1,
          `pocket ${plan.pocketIndex}`);
    const lifted = g.ballToCamera(g.pockets[plan.pocketIndex]);
    check('pot line is not the point lifted onto the cushion',
          Math.hypot(plan.pocket[0] - lifted[0], plan.pocket[1] - lifted[1]) > 2);
  }

  function pocketChoiceTest() {
    console.log('Pocket choice (top-down coordinates)');
    const g = new S.SnookerGeometry();
    g.updateHomography(g.dstPts);  // camera == top-down
    let plan = g.planShot([300, 1000], [200, 800], []);
    check('close middle pocket beats a far corner with a thinner cut', plan.pocketIndex === 4, `pocket ${plan.pocketIndex}, cut ${Math.round(plan.cutDeg)}`);
    plan = g.planShot([70, 150], [60, 400], []);
    check('steep angle into a middle pocket is not offered', !plan || plan.pocketIndex !== 4, plan ? `pocket ${plan.pocketIndex}` : 'none');
    const first = g.planShot([355, 900], [355, 500], []);
    const other = 1 - first.pocketIndex;
    const again = g.planShot([355, 900], [355, 500], [], other);
    check('near-equal pockets: the previous choice is kept', again.pocketIndex === other, `first ${first.pocketIndex}, kept ${again.pocketIndex}`);
  }

  function jitterTest() {
    console.log('Jitter on still balls');
    let seed = 1;
    const rand = () => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648; };
    const balls = [[200, 300, 'red'], [260, 340, 'red'], [400, 500, 'blue'], [520, 610, 'black'], [330, 700, 'white'], [600, 250, 'pink']];
    const tr = new S.BallTracker();
    let t = 0, movingAfterSettle = 0;
    const versions = [];
    for (let i = 0; i < 50; i++) {
      const r = 10;
      const dets = balls.map(([x, y, c]) => [x + (rand() * 0.6 - 0.3) * r, y + (rand() * 0.6 - 0.3) * r, r, c]);
      if (i % 7 === 3) dets[i % dets.length][0] += 0.8 * r;  // a one-frame outlier now and then
      tr.update(dets, t, () => false);
      if (t > 1.5 && !tr.settled(t)) movingAfterSettle++;
      versions.push(tr.version);
      t += 0.2;
    }
    check('jittering still balls never count as moving', movingAfterSettle === 0, `${movingAfterSettle} moving updates`);
    check("jitter doesn't replan every update", versions[49] - versions[10] < 20, `${versions[49] - versions[10]} replans in 8 s`);
  }

  function pocketLeatherTest() {
    console.log('Pocket leather (test_data/直播实测_主视角_球员遮挡.png)');
    const base = imread('test_data/直播实测_主视角_球员遮挡.png');
    for (const [name, rgb] of [['cream', [235, 225, 200]], ['off-white', [230, 225, 215]]]) {
      const im = base.clone();
      cv.circle(im, new cv.Point(326, 311), 8, new cv.Scalar(...rgb), -1);
      const a = new S.SnookerAnalyzer();
      let t = 0;
      for (let i = 0; i < 40; i++) { a.process(im, t); t += 0.05; }
      const balls = a.tracker.balls();
      const atPocket = balls.filter(b => Math.hypot(b.pos[0] - 326, b.pos[1] - 311) < 10).map(b => b.colour);
      const cue = a.cueBall, pink = balls.find(b => b.colour === 'pink');
      check(`${name} leather at a pocket: not a ball, cue ball and pink unaffected`,
            !atPocket.length && !!cue && Math.hypot(cue.pos[0] - 683, cue.pos[1] - 549) < 8 && !!pink && Math.hypot(pink.pos[0] - 519, pink.pos[1] - 356) < 8,
            `at pocket: ${atPocket}, cue: ${cue ? cue.pos.map(Math.round) : 'none'}`);
      im.delete();
    }
    base.delete();
  }

  function duplicateTest() {
    console.log('Duplicate balls');
    const neverCloth = () => false;  // the old spot is never bare: worst case for phantoms
    let tr = new S.BallTracker();
    ['white', 'white', 'white', 'brown', 'white', 'white', 'brown', 'brown', 'white'].forEach((c, i) =>
      tr.update([[683, 549, 8, c], [520, 569, 7, 'black']], i * 0.2, neverCloth));
    check('colour label flickering on one ball: still one ball', tr.balls().length === 2, tr.balls().map(t => t.colour).join(', '));
    tr = new S.BallTracker();
    let x = 683;
    for (let i = 0; i < 12; i++) {
      if (i >= 4 && i < 8) x += 14;  // rolls ~1.75 radii between detections, then stops
      tr.update([[x, 549, 8, 'white'], [520, 569, 7, 'black']], i * 0.2, neverCloth);
    }
    check('ball rolling to a stop leaves no phantom behind', tr.balls().length === 2, tr.balls().map(t => `${t.colour}@${Math.round(t.pos[0])}`).join(', '));
    const im = imread('test_data/直播实测_主视角_球员遮挡.png');
    const a = new S.SnookerAnalyzer();
    let t = 0;
    for (let i = 0; i < 40; i++) { a.process(im, t); t += 0.05; }
    im.delete();
    const plan = a.geometry.planShot([683, 549], [520, 569], [[683, 549], [687, 554], [520, 569]]);
    check("a double detection of the cue ball doesn't block its own path", !plan.blocked);
  }

  function cueCentreTest() {
    console.log('Cue ball centre');
    for (const [image, portrait, truth] of [['梯形桌面截屏.png', true, [1246.6, 995.7]], ['test_data/直播实测_主视角_球员遮挡.png', true, [683, 549]]]) {
      const im = imread(image);
      const v = new S.SnookerVision(), g = new S.SnookerGeometry();
      g.setOrientation(portrait);
      const { quad } = v.detectMainView(im);
      g.updateHomography(S.TableLock.sortCorners(quad));
      const whites = v.findBalls(im, g.playingArea(), (x, y) => g.ballRadiusPx(x, y)).filter(b => b[3] === 'white');
      const err = whites.length ? Math.hypot(whites[0][0] - truth[0], whites[0][1] - truth[1]) : null;
      check(`${image}: cue ball centre within 2 px`, err !== null && err < 2, err === null ? 'not found' : `${err.toFixed(1)} px off`);
      im.delete();
    }
  }

  const t0 = Date.now();
  cueCentreTest();
  duplicateTest();
  pocketMouthTest();
  pocketChoiceTest();
  jitterTest();
  pocketLeatherTest();
  streamTest();
  robustnessTest();
  palePinkTest();
  console.log(`\n${failures.length ? `${failures.length} failed` : 'All passed'}  (${((Date.now() - t0) / 1000).toFixed(1)} s)`);
  process.exit(failures.length ? 1 : 0);
})();
