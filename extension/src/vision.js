/*
 * Table and ball recognition. JavaScript port of vision.py (same logic and thresholds),
 * running on OpenCV.js. Frames are RGB cv.Mat (CV_8UC3); HSV/Lab values match the
 * Python version, which works on BGR with the BGR conversions.
 *
 * Memory: every cv.Mat created here is deleted here, except the table mask that
 * detectMainView() leaves in this.tableMask (owned by the vision object).
 */
(function (root) {
  'use strict';
  const S = root.Snooker = root.Snooker || {};

  const GREEN_LO = [35, 40, 40], GREEN_HI = [85, 255, 255];
  const WHITE_LO = [0, 0, 170], WHITE_HI = [180, 90, 255];

  // ---- small helpers ------------------------------------------------------

  function del(...mats) { for (const m of mats) if (m && !m.isDeleted()) m.delete(); }

  function toHSV(rgb) { const m = new cv.Mat(); cv.cvtColor(rgb, m, cv.COLOR_RGB2HSV); return m; }
  function toLab(rgb) { const m = new cv.Mat(); cv.cvtColor(rgb, m, cv.COLOR_RGB2Lab); return m; }

  /** Per-pixel range test on a 3-channel Mat -> CV_8UC1 mask (255 inside). */
  function inRange3(src, lo, hi) {
    const dst = new cv.Mat(src.rows, src.cols, cv.CV_8UC1);
    const s = src.data, d = dst.data;
    for (let i = 0, j = 0; j < d.length; i += 3, j++) {
      const a = s[i], b = s[i + 1], c = s[i + 2];
      d[j] = (a >= lo[0] && a <= hi[0] && b >= lo[1] && b <= hi[1] && c >= lo[2] && c <= hi[2]) ? 255 : 0;
    }
    return dst;
  }

  function andInPlace(a, b) { const x = a.data, y = b.data; for (let i = 0; i < x.length; i++) x[i] = x[i] & y[i]; }

  /** numpy-style median (mean of the two middle values for even counts). */
  function median(values) {
    const n = values.length;
    if (!n) return NaN;
    const s = Float64Array.from(values).sort();
    return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
  }

  /** Median of a uint8 channel over masked pixels, via a histogram. */
  function maskedMedian(src3, channel, mask) {
    const hist = new Uint32Array(256);
    const s = src3.data, m = mask.data;
    let n = 0;
    for (let j = 0, i = channel; j < m.length; j++, i += 3) if (m[j]) { hist[s[i]]++; n++; }
    if (!n) return NaN;
    const kth = k => { let acc = 0; for (let v = 0; v < 256; v++) { acc += hist[v]; if (acc > k) return v; } return 255; };
    return n % 2 ? kth((n - 1) / 2) : (kth(n / 2 - 1) + kth(n / 2)) / 2;
  }

  function polyMat(points) {
    const flat = [];
    for (const [x, y] of points) flat.push(Math.trunc(x), Math.trunc(y));  // like np.int32
    return cv.matFromArray(points.length, 1, cv.CV_32SC2, flat);
  }

  function fillPolyMask(rows, cols, points) {
    const mask = cv.Mat.zeros(rows, cols, cv.CV_8UC1);
    const pts = polyMat(points), vec = new cv.MatVector();
    vec.push_back(pts);
    cv.fillPoly(mask, vec, new cv.Scalar(255));
    vec.delete(); pts.delete();
    return mask;
  }

  function contoursOf(mask) {
    const contours = new cv.MatVector(), hier = new cv.Mat();
    cv.findContours(mask, contours, hier, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
    hier.delete();
    const list = [];
    for (let i = 0; i < contours.size(); i++) list.push(contours.get(i));
    contours.delete();
    return list;  // caller deletes each
  }

  function morphOpen(mask, kernel) {
    const out = new cv.Mat();
    cv.morphologyEx(mask, out, cv.MORPH_OPEN, kernel);
    return out;
  }

  /** Convex polygon grown outwards by d pixels (each side pushed out, neighbours intersected). */
  function expandPolygon(poly, d) {
    const n = poly.length;
    let area2 = 0;
    for (let i = 0; i < n; i++) area2 += poly[i][0] * poly[(i + 1) % n][1] - poly[(i + 1) % n][0] * poly[i][1];
    const sign = area2 > 0 ? 1 : -1;
    const lines = poly.map((p, i) => {
      const q = poly[(i + 1) % n], e = [q[0] - p[0], q[1] - p[1]];
      const len = Math.max(Math.hypot(e[0], e[1]), 1e-9);
      const nx = sign * e[1] / len, ny = -sign * e[0] / len;   // points outwards
      return [[p[0] + nx * d, p[1] + ny * d], e];
    });
    return lines.map((_, i) => {
      const [p1, d1] = lines[(i - 1 + n) % n], [p2, d2] = lines[i];
      const denom = d1[0] * d2[1] - d1[1] * d2[0];
      if (Math.abs(denom) < 1e-9) return p2;
      const t = ((p2[0] - p1[0]) * d2[1] - (p2[1] - p1[1]) * d2[0]) / denom;
      return [p1[0] + t * d1[0], p1[1] + t * d1[1]];
    });
  }

  const hypot = Math.hypot;

  // ---- SnookerVision ------------------------------------------------------

  class SnookerVision {
    constructor() {
      this.tableMask = null;   // CV_8UC1 table area (may point at a TableLock view's mask)
      this._ownMask = null;    // the mask this object allocated and must free
    }

    _setOwnMask(m) { del(this._ownMask); this._ownMask = m; this.tableMask = m; }

    /** Returns {ok, quad} where quad is 4 [x, y] corners (any order) of the full table. */
    detectMainView(frame) {
      this._setOwnMask(null);
      const hsv = toHSV(frame);
      const mask = inRange3(hsv, GREEN_LO, GREEN_HI);
      hsv.delete();
      const total = mask.rows * mask.cols;
      // The table can be small when the capture covers a whole web page around the player
      if (cv.countNonZero(mask) / total < SnookerVision.MIN_TABLE_AREA) { mask.delete(); return { ok: false }; }

      const contours = contoursOf(mask);
      let best = null, bestArea = -1;
      for (const c of contours) { const a = cv.contourArea(c); if (a > bestArea) { bestArea = a; best = c; } }
      const result = (() => {
        if (!best || bestArea < total * SnookerVision.MIN_TABLE_AREA) return { ok: false };
        // Balls are holes in the green contour, so use the convex hull
        const hull = new cv.Mat();
        cv.convexHull(best, hull, false, true);
        const quad = SnookerVision.fitQuad(hull, frame.rows, frame.cols);
        hull.delete();
        if (!quad) return { ok: false };
        // The table area is the corner quadrilateral, which also covers a corner hidden by a player
        this._setOwnMask(fillPolyMask(frame.rows, frame.cols, quad));
        return { ok: true, quad };
      })();
      for (const c of contours) c.delete();
      mask.delete();
      return result;
    }

    /**
     * Four table corners from the outline even when a player or caption hides one:
     * the four longest sides of the finely simplified outline, neighbours intersected.
     */
    static fitQuad(hull, rows, cols) {
      const peri = cv.arcLength(hull, true);
      const approx = new cv.Mat();
      cv.approxPolyDP(hull, approx, 0.01 * peri, true);
      const d = approx.data32S, poly = [];
      for (let i = 0; i < d.length; i += 2) poly.push([d[i], d[i + 1]]);
      approx.delete();
      const n = poly.length;
      if (n < 4 || n > 12) return null;
      if (n === 4) return poly;

      const sides = poly.map((p, i) => [hypot(poly[(i + 1) % n][0] - p[0], poly[(i + 1) % n][1] - p[1]), i]);
      const keep = sides.slice().sort((a, b) => b[0] - a[0] || b[1] - a[1]).slice(0, 4).map(s => s[1]).sort((a, b) => a - b);
      const lines = keep.map(i => [poly[i], poly[(i + 1) % n]]);
      const corners = [];
      for (let k = 0; k < 4; k++) {
        const [p1, p2] = lines[k], [p3, p4] = lines[(k + 1) % 4];
        const d1 = [p2[0] - p1[0], p2[1] - p1[1]], d2 = [p4[0] - p3[0], p4[1] - p3[1]];
        const denom = d1[0] * d2[1] - d1[1] * d2[0];
        if (Math.abs(denom) < 1e-6) return null;
        const t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / denom;
        corners.push([p1[0] + t * d1[0], p1[1] + t * d1[1]]);
      }
      for (const [x, y] of corners) {
        if (x < -0.1 * cols || x > 1.1 * cols || y < -0.1 * rows || y > 1.1 * rows) return null;
      }
      const cm = cv.matFromArray(4, 1, cv.CV_32FC2, corners.flat());
      const convex = cv.isContourConvex(cm);
      cm.delete();
      if (!convex) return null;
      return corners.map(([x, y]) => [Math.round(x), Math.round(y)]);
    }

    /** Largest roughly-circular bright, low-saturation blob in the table area -> [x, y, r] or null. */
    detectCueBall(frame) {
      if (!this.tableMask) return null;
      const hsv = toHSV(frame);
      const raw = inRange3(hsv, WHITE_LO, WHITE_HI);
      hsv.delete();
      andInPlace(raw, this.tableMask);
      const kernel = cv.Mat.ones(3, 3, cv.CV_8U);
      const mask = morphOpen(raw, kernel);
      del(raw, kernel);
      const contours = contoursOf(mask);
      mask.delete();
      let best = null;
      for (const c of contours) {
        const area = cv.contourArea(c);
        if (area >= 20) {
          const { center, radius } = cv.minEnclosingCircle(c);
          if (area / (Math.PI * radius * radius) >= 0.45 && (!best || area > best[0])) best = [area, center.x, center.y, radius];
        }
        c.delete();
      }
      return best ? best.slice(1) : null;
    }

    /** Every ball on the playing surface -> list of [x, y, r]. */
    detectBalls(frame, innerPoly, radiusAt) {
      const h = frame.rows, w = frame.cols;
      const radii = innerPoly.map(([x, y]) => radiusAt(x, y));
      const rMin = Math.trunc(Math.min(...radii) * 0.6), rMax = Math.trunc(Math.max(...radii) * 1.4);
      if (rMin < 3) return [];

      // Edges away from the table (web page text, crowd, captions) can't be balls and make
      // circle detection slow, so they are blanked out of the edge map below
      const play = fillPolyMask(h, w, innerPoly);
      const nearPlay = fillPolyMask(h, w, expandPolygon(innerPoly, rMax + 2));

      // Compare against the local cloth colour in Lab (the green ball is close to the cloth in hue)
      const lab = toLab(frame);
      const small = new cv.Mat();
      cv.resize(lab, small, new cv.Size(Math.floor(w / 4), Math.floor(h / 4)), 0, 0, cv.INTER_AREA);
      const k = Math.max(Math.floor(w / 4 / 37) | 1, 5);
      cv.medianBlur(small, small, k);
      const cloth = new cv.Mat();
      cv.resize(small, cloth, new cv.Size(w, h), 0, 0, cv.INTER_LINEAR);
      small.delete();
      const diff = new cv.Mat(h, w, cv.CV_8UC1);
      {
        const a = lab.data, b = cloth.data, d = diff.data, n = nearPlay.data;
        for (let i = 0, j = 0; j < d.length; i += 3, j++) {
          if (!n[j]) { d[j] = 0; continue; }
          const x = a[i] - b[i], y = a[i + 1] - b[i + 1], z = a[i + 2] - b[i + 2];
          d[j] = Math.min(255, Math.sqrt(x * x + y * y + z * z) * 3);
        }
      }
      del(lab, cloth, nearPlay);

      const blurred = new cv.Mat();
      cv.GaussianBlur(diff, blurred, new cv.Size(5, 5), 1.5);
      diff.delete();
      const circles = new cv.Mat();
      cv.HoughCircles(blurred, circles, cv.HOUGH_GRADIENT_ALT, 1.5, rMin * 1.5, 150, 0.6, rMin, rMax);
      blurred.delete();

      const pd = play.data, cd = circles.data32F;
      const balls = [];
      for (let i = 0; i + 2 < cd.length; i += 3) {
        const x = cd[i], y = cd[i + 1], r = cd[i + 2];
        const xi = Math.trunc(x), yi = Math.trunc(y);
        if (!(yi >= 0 && yi < h && xi >= 0 && xi < w) || pd[yi * w + xi] === 0) continue;
        // Reject circles that aren't ball-sized for this spot (gaps between clustered reds, chalk marks)
        const ratio = r / radiusAt(x, y);
        if (ratio < 0.7 || ratio > 1.35) continue;
        balls.push([x, y, r]);
      }
      del(play, circles);
      return balls;
    }

    /**
     * Full ball pass: detection + colour labels, robust to brightness and resolution.
     * Returns a list of [x, y, r, colour] in frame pixels. Uses this.tableMask.
     */
    findBalls(frame, innerPoly, radiusAt, exclude = null) {
      if (!this.tableMask) return [];
      const tableMask = this.tableMask;
      const owned = [];   // temporary mats to free at the end

      // Brightness: scale the frame so the cloth always has about the same brightness
      const hsv = toHSV(frame);
      const cloth = inRange3(hsv, GREEN_LO, GREEN_HI);
      andInPlace(cloth, tableMask);
      let clothHue = null, clothSat = null;
      if (cv.countNonZero(cloth) > 100) {
        clothHue = maskedMedian(hsv, 0, cloth);
        clothSat = maskedMedian(hsv, 1, cloth);
        const gain = Math.min(2.0, Math.max(0.6, SnookerVision.CLOTH_BRIGHTNESS / Math.max(maskedMedian(hsv, 2, cloth), 1)));
        if (Math.abs(gain - 1) > 0.05) {
          const f = new cv.Mat();
          cv.convertScaleAbs(frame, f, gain, 0);
          owned.push(f); frame = f;
        }
      }
      del(hsv, cloth);

      // Resolution: small balls are unreliable for circle detection, so upscale first
      const smallest = Math.min(...innerPoly.map(([x, y]) => radiusAt(x, y)));
      let k = Math.min(3.0, Math.max(1.0, SnookerVision.MIN_DETECT_RADIUS / Math.max(smallest, 1)));
      if (k > 1.05) {
        const f = new cv.Mat();
        cv.resize(frame, f, new cv.Size(0, 0), k, k, cv.INTER_CUBIC);
        owned.push(f); frame = f;
        const m = new cv.Mat();
        cv.resize(tableMask, m, new cv.Size(frame.cols, frame.rows), 0, 0, cv.INTER_NEAREST);
        owned.push(m); this.tableMask = m;
      } else {
        k = 1.0;
      }
      let balls, labels, occluded;
      try {
        const poly = innerPoly.map(([x, y]) => [x * k, y * k]);
        const rAt = (x, y) => radiusAt(x / k, y / k) * k;
        balls = this.detectBalls(frame, poly, rAt);
        const cue = this.detectCueBall(frame);
        if (cue && balls.every(b => hypot(b[0] - cue[0], b[1] - cue[1]) > cue[2])) {
          // The bright patch is only the lit top of the ball: fit the whole outline instead
          balls.push(this.refineCircle(frame, cue[0], cue[1], rAt(cue[0], cue[1])) || cue);
        }
        // Spots that can't hold a ball (pocket mouths) go before colours are assigned, so a
        // look-alike there can't take the one-of-each colour slot from a real ball
        if (exclude) balls = balls.filter(b => !exclude(b[0] / k, b[1] / k));
        labels = this.classifyBalls(frame, balls, clothHue);
        if (!labels.includes('green') && clothHue !== null) {
          const green = this.findGreenBall(frame, poly, rAt, clothHue, clothSat, balls);
          if (green && !(exclude && exclude(green[0] / k, green[1] / k))) { balls.push(green); labels.push('green'); }
        }
        occluded = this.occluderMask(frame, smallest * k);
      } finally {
        this.tableMask = tableMask;
        del(...owned);
      }
      const out = [];
      balls.forEach(([x, y, r], i) => {
        const label = labels[i];
        if (label === 'cloth') return;
        // An unrecognisable "ball" inside a player's body or arm reaching over the table
        if (label === 'unknown') {
          const yi = Math.min(Math.trunc(y), occluded.rows - 1), xi = Math.min(Math.trunc(x), occluded.cols - 1);
          if (occluded.data[yi * occluded.cols + xi]) return;
        }
        out.push([x / k, y / k, r / k, label]);
      });
      return out;
    }

    /**
     * Areas covered by something reaching in over the table edge (a player, an arm):
     * big non-cloth regions touching the table edge. Thin cues are opened away first.
     * Returns {rows, cols, data: Uint8Array} (1 = covered).
     */
    occluderMask(frame, ballR) {
      const hsv = toHSV(frame);
      const nonCloth = inRange3(hsv, GREEN_LO, GREEN_HI);
      hsv.delete();
      {
        const d = nonCloth.data, t = this.tableMask.data;
        for (let i = 0; i < d.length; i++) d[i] = (255 - d[i]) & t[i];
      }
      const size = Math.max(Math.trunc(ballR * 0.8) | 1, 3);
      const kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(size, size));
      const opened = morphOpen(nonCloth, kernel);
      del(nonCloth, kernel);
      const k7 = cv.Mat.ones(7, 7, cv.CV_8U), eroded = new cv.Mat();
      cv.erode(this.tableMask, eroded, k7);
      const labels = new cv.Mat(), stats = new cv.Mat(), cents = new cv.Mat();
      const n = cv.connectedComponentsWithStats(opened, labels, stats, cents);
      const minArea = 6 * Math.PI * ballR * ballR;
      const touches = new Uint8Array(n);
      const L = labels.data32S, T = this.tableMask.data, E = eroded.data;
      for (let i = 0; i < L.length; i++) if (L[i] && T[i] && !E[i]) touches[L[i]] = 1;
      const big = new Uint8Array(n);
      const st = stats.data32S;
      for (let i = 1; i < n; i++) big[i] = touches[i] && st[i * 5 + cv.CC_STAT_AREA] >= minArea ? 1 : 0;
      const data = new Uint8Array(L.length);
      for (let i = 0; i < L.length; i++) data[i] = big[L[i]];
      const out = { rows: opened.rows, cols: opened.cols, data };
      del(opened, k7, eroded, labels, stats, cents);
      return out;
    }

    /**
     * Second look for the green ball (faint edge: blur, small size, compression). It is either
     * clearly bluer than the cloth, or only a little bluer but greyer, depending on the broadcast.
     */
    findGreenBall(frame, innerPoly, radiusAt, clothHue, clothSat, known) {
      const h = frame.rows, w = frame.cols;
      const hsv = toHSV(frame);
      const raw = new cv.Mat(h, w, cv.CV_8UC1);
      {
        const s = hsv.data, d = raw.data;
        for (let i = 0, j = 0; j < d.length; i += 3, j++) {
          const hue = s[i], sat = s[i + 1], val = s[i + 2];
          const bluer = hue >= clothHue + 8;
          const greyer = hue >= clothHue + 4 && sat <= 0.9 * clothSat;
          d[j] = ((bluer || greyer) && hue <= 95 && sat >= 60 && val >= 45) ? 255 : 0;
        }
      }
      hsv.delete();
      const play = fillPolyMask(h, w, innerPoly);
      andInPlace(raw, play);
      play.delete();
      const kernel = cv.Mat.ones(3, 3, cv.CV_8U);
      const mask = morphOpen(raw, kernel);
      del(raw, kernel);
      const contours = contoursOf(mask);
      mask.delete();
      let best = null;
      for (const c of contours) {
        const { center, radius: r } = cv.minEnclosingCircle(c);
        const x = center.x, y = center.y, rExp = radiusAt(x, y), area = cv.contourArea(c);
        c.delete();
        // Roughly ball-sized (the shaded half may be lost) and roundish
        if (r < 0.4 * rExp || r > 1.4 * rExp || area < 0.2 * Math.PI * rExp * rExp) continue;
        if (area / (Math.PI * r * r) < 0.45) continue;
        if (known.some(b => hypot(x - b[0], y - b[1]) < b[2])) continue;
        if (!best || area > best[0]) best = [area, x, y, rExp];
      }
      return best ? best.slice(1) : null;
    }

    /**
     * Fits the whole outline of the ball near (x, y): a ball's lower half is in shade, so a
     * bright-patch detection sits too high and too small. Looks for a ball-sized circle in the colour
     * difference from the surrounding cloth (upscaled first if the ball is small). Returns [x, y, r] or null.
     */
    refineCircle(frame, x, y, rExp) {
      const h = frame.rows, w = frame.cols;
      const R = Math.trunc(2.5 * rExp) + 2;
      const x0 = Math.max(Math.trunc(x) - R, 0), y0 = Math.max(Math.trunc(y) - R, 0);
      const x1 = Math.min(Math.trunc(x) + R, w), y1 = Math.min(Math.trunc(y) + R, h);
      if (x1 - x0 < 8 || y1 - y0 < 8) return null;
      const view = frame.roi(new cv.Rect(x0, y0, x1 - x0, y1 - y0));
      const k = Math.max(1, 14 / rExp);
      let roi;
      if (k > 1) { roi = new cv.Mat(); cv.resize(view, roi, new cv.Size(0, 0), k, k, cv.INTER_CUBIC); }
      else roi = view.clone();
      view.delete();
      const rk = rExp * k;
      const hsv = toHSV(roi), lab = toLab(roi);
      const cloth = inRange3(hsv, GREEN_LO, GREEN_HI);
      hsv.delete();
      if (cv.countNonZero(cloth) < 10) { del(roi, lab, cloth); return null; }
      const ref = [0, 1, 2].map(c => maskedMedian(lab, c, cloth));
      const diff = new cv.Mat(roi.rows, roi.cols, cv.CV_8UC1);
      {
        const a = lab.data, d = diff.data;
        for (let i = 0, j = 0; j < d.length; i += 3, j++) {
          const p = a[i] - ref[0], q = a[i + 1] - ref[1], s = a[i + 2] - ref[2];
          d[j] = Math.min(255, Math.sqrt(p * p + q * q + s * s) * 3);
        }
      }
      del(roi, lab, cloth);
      const blurred = new cv.Mat(), circles = new cv.Mat();
      cv.GaussianBlur(diff, blurred, new cv.Size(5, 5), 1.5);
      cv.HoughCircles(blurred, circles, cv.HOUGH_GRADIENT_ALT, 1.5, rk, 150, 0.4, Math.trunc(rk * 0.75), Math.ceil(rk * 1.25));
      del(diff, blurred);
      const cd = circles.data32F, cx = (x - x0) * k, cy = (y - y0) * k;
      let best = null;
      for (let i = 0; i + 2 < cd.length; i += 3) {
        const d = hypot(cd[i] - cx, cd[i + 1] - cy);
        if (!best || d < best[0]) best = [d, cd[i], cd[i + 1], cd[i + 2]];
      }
      circles.delete();
      if (!best || best[0] > 0.8 * rk) return null;
      return [best[1] / k + x0, best[2] / k + y0, best[3] / k];
    }

    /** True if the ball-sized disc at (x, y) is almost entirely bare cloth. */
    isClothAt(frame, x, y, r) {
      const h = frame.rows, w = frame.cols;
      const rr = Math.max(Math.trunc(r * 0.8), 2);
      const x0 = Math.max(Math.trunc(x) - rr, 0), y0 = Math.max(Math.trunc(y) - rr, 0);
      const x1 = Math.min(Math.trunc(x) + rr + 1, w), y1 = Math.min(Math.trunc(y) + rr + 1, h);
      if (x1 <= x0 || y1 <= y0) return false;
      const roi = frame.roi(new cv.Rect(x0, y0, x1 - x0, y1 - y0));
      const hsv = toHSV(roi);
      const g = inRange3(hsv, GREEN_LO, GREEN_HI);
      const frac = cv.countNonZero(g) / (g.rows * g.cols);
      del(roi, hsv, g);
      return frac > 0.85;
    }

    /** Median HSV and Lab of the ball's inner disc, ignoring specular highlights. */
    ballColorFeatures(frame, x, y, r) {
      const h = frame.rows, w = frame.cols;
      const rr = Math.max(Math.trunc(r * 0.6), 2);
      const x0 = Math.max(Math.trunc(x) - rr, 0), y0 = Math.max(Math.trunc(y) - rr, 0);
      const x1 = Math.min(Math.trunc(x) + rr + 1, w), y1 = Math.min(Math.trunc(y) + rr + 1, h);
      if (x1 <= x0 || y1 <= y0) return null;
      const roi = frame.roi(new cv.Rect(x0, y0, x1 - x0, y1 - y0));
      const patch = roi.clone();
      roi.delete();
      const disc = cv.Mat.zeros(patch.rows, patch.cols, cv.CV_8UC1);
      cv.circle(disc, new cv.Point(Math.trunc(x) - x0, Math.trunc(y) - y0), rr, new cv.Scalar(255), -1);
      const hsv = toHSV(patch), lab = toLab(patch);
      const H = hsv.data, Lb = lab.data, D = disc.data;
      let pix = [];
      for (let j = 0; j < D.length; j++) {
        if (D[j]) pix.push([H[3 * j], H[3 * j + 1], H[3 * j + 2], Lb[3 * j], Lb[3 * j + 1], Lb[3 * j + 2]]);
      }
      del(patch, disc, hsv, lab);
      const keep = pix.filter(p => !(p[2] > 220 && p[1] < 70));
      if (keep.length > 5) pix = keep;
      const med = c => median(pix.map(p => p[c]));
      return { hsv: [med(0), med(1), med(2)], lab: [med(3), med(4), med(5)] };
    }

    /**
     * Colour labels: red, yellow, green, brown, blue, pink, black, white, unknown, or "cloth"
     * (not a ball). Red/pink/brown are judged relative to the median red-ish ball; only one
     * ball of each colour exists, so duplicates keep the strongest candidate.
     */
    classifyBalls(frame, balls, clothHue = null) {
      const feats = balls.map(b => this.ballColorFeatures(frame, b[0], b[1], b[2]));
      const labels = [], scores = [], reddish = [];
      feats.forEach((f, i) => {
        if (!f) { labels.push('unknown'); scores.push(0); return; }
        const [hh, ss, vv] = f.hsv, [L, A] = f.lab;
        let label = 'unknown', score = 0;
        if (clothHue !== null && vv < 45 && ss > 230 && Math.abs(hh - clothHue) < 12) {
          label = 'cloth';                  // deep-green darkness: shadow between clustered balls
        } else if (vv < 45) {
          label = 'black'; score = 45 - vv;
        } else if (L > 150 && ss < 120) {
          // Bright and pale: the cue ball, or a glossy pink washed out by strong lighting. The white
          // is neutral-to-cream (Lab a* 110-124; an over-exposed white reads 128 but has no colour),
          // while a pink keeps a red tint however pale it gets.
          if (A < 126 || (A < 130 && ss < 12)) { label = 'white'; score = (130 - A) * 4 + (L - ss) / 10; }  // least red wins
          else { label = 'reddish'; reddish.push(i); }
        } else if (ss < 60 && vv < 150) {
          label = 'cloth';                  // dark grey: no ball is grey (clothing, cue, shadowed hand)
        } else if (hh >= 95 && hh <= 130) {
          label = 'blue'; score = ss;
        } else if (clothHue !== null && Math.abs(hh - clothHue) < 8) {
          label = 'cloth';
        } else if (hh >= 50 && hh < 95) {
          label = 'green'; score = ss;
        } else if (hh >= 16 && hh < 40 && vv > 170) {
          label = 'yellow'; score = vv;
        } else if (hh < 40 || hh >= 140) {
          label = 'reddish'; reddish.push(i);  // includes dull orange/olive: the brown
        }
        labels.push(label); scores.push(score);
      });

      let ref = SnookerVision.DEFAULT_RED_LAB;
      if (reddish.length >= 4) ref = [0, 1, 2].map(c => median(reddish.map(i => feats[i].lab[c])));
      for (const i of reddish) {
        const [L, A, B] = feats[i].lab;
        const pinkScore = (ref[2] - B) + 0.3 * (L - ref[0]);
        const brownScore = ref[1] - A;
        if (ref[2] - B > 10 && L > ref[0]) { labels[i] = 'pink'; scores[i] = pinkScore; }
        else if (brownScore > 10) { labels[i] = 'brown'; scores[i] = brownScore; }
        else { labels[i] = 'red'; scores[i] = 0; }
      }

      for (const colour of ['white', 'yellow', 'green', 'brown', 'blue', 'pink', 'black']) {
        const idx = labels.map((l, i) => l === colour ? i : -1).filter(i => i >= 0);
        if (idx.length > 1) {
          const best = idx.reduce((a, b) => scores[b] > scores[a] ? b : a);
          for (const i of idx) if (i !== best) labels[i] = (colour === 'pink' || colour === 'brown') ? 'red' : 'unknown';
        }
      }
      return labels;
    }
  }
  SnookerVision.MIN_TABLE_AREA = 0.08;
  SnookerVision.CLOTH_BRIGHTNESS = 145.0;
  SnookerVision.MIN_DETECT_RADIUS = 11.0;
  SnookerVision.DEFAULT_RED_LAB = [90.0, 178.0, 165.0];

  // ---- TableLock ------------------------------------------------------------

  /**
   * Remembers the camera views that show the whole table (up to maxViews). A view is
   * learned when the same full-table outline holds steady for a few frames (corners =
   * median over them); later frames match a known view by overlap of the table area.
   */
  class TableLock {
    constructor(framesNeeded = 8, iouThreshold = 0.8, maxViews = 4) {
      this.framesNeeded = framesNeeded;
      this.iouThreshold = iouThreshold;
      this.maxViews = maxViews;
      this.views = [];
      this.reset();
    }

    reset() {
      for (const v of this.views || []) del(v.fullMask);
      this.views = [];       // {corners, fullMask (cv.Mat), small (Uint8Array)}
      this.active = null;
      this._candidates = [];
    }

    get locked() { return this.views.length > 0; }
    get corners() { return this.active !== null ? this.views[this.active].corners : null; }
    get mask() { return this.active !== null ? this.views[this.active].fullMask : null; }

    static small(mask) {
      const m = new cv.Mat();
      cv.resize(mask, m, new cv.Size(160, 90), 0, 0, cv.INTER_NEAREST);
      const out = Uint8Array.from(m.data, v => v > 0 ? 1 : 0);
      m.delete();
      return out;
    }

    _addView(corners, rows, cols) {
      // Use the corner quadrilateral as the table area: the outline in this frame may lack a hidden corner
      const fullMask = fillPolyMask(rows, cols, corners.map(([x, y]) => [Math.round(x), Math.round(y)]));
      if (this.views.length >= this.maxViews) del(this.views.shift().fullMask);
      this.views.push({ corners, fullMask, small: TableLock.small(fullMask) });
      this.active = this.views.length - 1;
    }

    lockManual(corners, rows, cols) {
      this.reset();
      this._addView(corners.map(([x, y]) => [x, y]), rows, cols);
    }

    static sortCorners(quad) {
      const pts = quad.slice().sort((a, b) => a[1] - b[1]);
      const top = pts.slice(0, 2).sort((a, b) => a[0] - b[0]);
      const bottom = pts.slice(2).sort((a, b) => b[0] - a[0]);
      return [...top, ...bottom].map(([x, y]) => [x, y]);
    }

    _plausibleFullTable(corners, rows, cols, portrait) {
      if (!portrait) return true;
      const [tl, tr, br, bl] = corners;
      // Broadcast views look down the table: a trapezoid narrowing towards the far end
      if (hypot(tr[0] - tl[0], tr[1] - tl[1]) > 0.9 * hypot(br[0] - bl[0], br[1] - bl[1])) return false;
      // A table cut off by the frame edge (zoomed-in shot) has fake corners
      const margin = 0.01 * Math.max(rows, cols);
      return corners.every(([x, y]) => x >= margin && y >= margin && x <= cols - margin && y <= rows - margin);
    }

    /** Feed one frame's detection. Returns true if the frame shows a known full-table view. */
    observe(isTable, quad, tableMask, portrait) {
      this.active = null;
      if (!isTable || !tableMask) { this._candidates = []; return false; }

      const small = TableLock.small(tableMask);
      let best = null, bestIou = 0;
      this.views.forEach((v, i) => {
        if (v.fullMask.rows !== tableMask.rows || v.fullMask.cols !== tableMask.cols) return;
        let inter = 0, union = 0;
        for (let j = 0; j < small.length; j++) { const a = v.small[j], b = small[j]; inter += a & b; union += a | b; }
        const iou = union ? inter / union : 0;
        if (iou > bestIou) { best = i; bestIou = iou; }
      });
      if (best !== null && bestIou >= this.iouThreshold) { this.active = best; this._candidates = []; return true; }

      if (!quad || quad.length !== 4) { this._candidates = []; return false; }
      const corners = TableLock.sortCorners(quad);
      if (!this._plausibleFullTable(corners, tableMask.rows, tableMask.cols, portrait)) { this._candidates = []; return false; }
      const diag = hypot(tableMask.rows, tableMask.cols);
      const last = this._candidates[this._candidates.length - 1];
      if (last && Math.max(...corners.map((c, i) => hypot(c[0] - last[i][0], c[1] - last[i][1]))) > 0.015 * diag) {
        this._candidates = [];  // camera still moving, start over
      }
      this._candidates.push(corners);
      if (this._candidates.length < this.framesNeeded) return false;
      const med = [0, 1, 2, 3].map(i => [median(this._candidates.map(c => c[i][0])), median(this._candidates.map(c => c[i][1]))]);
      this._addView(med, tableMask.rows, tableMask.cols);
      this._candidates = [];
      return true;
    }
  }

  S.SnookerVision = SnookerVision;
  S.TableLock = TableLock;
  S.median = median;
  S.expandPolygon = expandPolygon;
})(typeof self !== 'undefined' ? self : globalThis);
