/*
 * Table geometry and shot planning. JavaScript port of geometry.py.
 * Points are [x, y] arrays. "Top-down" coordinates are millimetres x scale on a
 * flat table; "camera" coordinates are frame pixels.
 */
(function (root) {
  'use strict';
  const S = root.Snooker = root.Snooker || {};

  const sub = (a, b) => [a[0] - b[0], a[1] - b[1]];
  const add = (a, b) => [a[0] + b[0], a[1] + b[1]];
  const mul = (a, k) => [a[0] * k, a[1] * k];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1];
  const norm = a => Math.hypot(a[0], a[1]);

  function applyH(M, p) {
    const x = p[0], y = p[1];
    const w = M[6] * x + M[7] * y + M[8];
    return [(M[0] * x + M[1] * y + M[2]) / w, (M[3] * x + M[4] * y + M[5]) / w];
  }

  function perspective(src, dst) {
    const a = cv.matFromArray(4, 1, cv.CV_32FC2, src.flat());
    const b = cv.matFromArray(4, 1, cv.CV_32FC2, dst.flat());
    const M = cv.getPerspectiveTransform(a, b);
    const out = Array.from(M.data64F);
    a.delete(); b.delete(); M.delete();
    return out;
  }

  class SnookerGeometry {
    constructor() {
      this.TABLE_WIDTH = 3569;   // playing area, mm
      this.TABLE_HEIGHT = 1778;
      // The corners found in the picture are the outer edge of the green cushion tops; the cushion
      // nose (edge of the playing area, where the pocket mouths are) is this far in
      this.CUSHION = 45;
      this.scale = 0.4;
      this.portrait = true;
      this.M = null;
      this.Minv = null;
      this._updateDst();
    }

    _updateDst() {
      // The main broadcast camera looks down the length of the table: TL->TR is a short side
      const long = Math.trunc(this.TABLE_WIDTH * this.scale), short = Math.trunc(this.TABLE_HEIGHT * this.scale);
      [this.targetW, this.targetH] = this.portrait ? [short, long] : [long, short];
      const w = this.targetW, h = this.targetH;
      this.dstPts = [[0, 0], [w, 0], [w, h], [0, h]];
      // Cushion-nose corners and the middle of each long-side nose. These are the jaws,
      // still on the cloth: the hole a ball falls into is further out (see pocketMouth).
      const m = this.CUSHION * this.scale;
      const middles = this.portrait ? [[m, h / 2], [w - m, h / 2]] : [[w / 2, m], [w / 2, h - m]];
      this.pockets = [[m, m], [w - m, m], [w - m, h - m], [m, h - m], ...middles];
    }

    setOrientation(portrait, srcPts = null) {
      this.portrait = portrait;
      this._updateDst();
      if (srcPts) this.updateHomography(srcPts);
    }

    get ballRadius() { return 52.5 / 2 * this.scale; }

    updateHomography(srcPts) {
      if (!srcPts || srcPts.length !== 4) return false;
      this.M = perspective(srcPts, this.dstPts);
      this.Minv = perspective(this.dstPts, srcPts);
      return true;
    }

    clearHomography() { this.M = null; this.Minv = null; }

    toTopDown(p) { return this.M ? applyH(this.M, p) : null; }
    toCamera(p) { return this.Minv ? applyH(this.Minv, p) : null; }

    /** Camera-view polygon of the playing area (inside the cushion noses). */
    playingArea() {
      const m = this.CUSHION * this.scale, w = this.targetW, h = this.targetH;
      return [[m, m], [w - m, m], [w - m, h - m], [m, h - m]].map(p => this.toCamera(p));
    }

    /**
     * True if a camera-view point sits right on a pocket mouth. No resting ball can be centred there,
     * but the cream pocket leather of some venues looks just like a ball. A ball against both
     * cushions in a corner is ~1.41 radii from the corner mouth; against a cushion, 1 radius from a middle one.
     */
    atPocketMouth(x, y) {
      const p = this.toTopDown([x, y]);
      return this.pockets.some((pocket, i) => norm(sub(p, pocket)) < (i < 4 ? 1.3 : 0.9) * this.ballRadius);
    }

    /**
     * How hard it is to pot target into pocket `index` from cue (top-down), or null if it can't go:
     * aiming error grows with both distances and with 1 / cos(cut); a middle pocket's opening narrows
     * as the ball comes in at an angle, and too steep an angle can't drop. Returns [difficulty, cutDeg].
     */
    pocketDifficulty(cue, target, index, maxCutDeg = 75) {
      const pocket = this.pocketMouth(index);
      const shot = sub(target, cue), pot = sub(pocket, target);
      const d1 = norm(shot), d2 = norm(pot);
      if (d1 === 0 || d2 === 0) return null;
      const cut = Math.acos(Math.min(1, Math.max(-1, dot(shot, pot) / (d1 * d2)))) * 180 / Math.PI;
      if (cut > maxCutDeg) return null;
      let width = 1.0;
      if (index >= 4) {  // middle pockets: compare with the straight-in direction (across the table)
        const straightIn = this.portrait ? [pocket[0] < this.targetW / 2 ? -1 : 1, 0] : [0, pocket[1] < this.targetH / 2 ? -1 : 1];
        const entry = Math.acos(Math.min(1, Math.max(-1, dot(mul(pot, 1 / d2), straightIn)))) * 180 / Math.PI;
        if (entry > SnookerGeometry.MIDDLE_POCKET_MAX_ENTRY_DEG) return null;
        width = Math.cos(entry * Math.PI / 180);
      }
      return [(d1 + this.ballRadius) * d2 / (Math.cos(cut * Math.PI / 180) * width), cut];
    }

    /**
     * Screen distance (downwards) from a ball's visible centre to where it touches the cloth. The
     * table mapping is for the cloth plane while a ball's centre is one radius above it; from a camera
     * at elevation e that radius shows as r*cos(e) pixels, and sin(e) is how much the table is
     * foreshortened along its length here (no drop at all in an overhead shot).
     */
    ballDropPx(x, y) {
      const c = this.toTopDown([x, y]), o = this.toCamera(c);
      const jx = norm(sub(this.toCamera([c[0] + 1, c[1]]), o)), jy = norm(sub(this.toCamera([c[0], c[1] + 1]), o));
      const sinE = Math.min(jx, jy) / Math.max(jx, jy, 1e-9);
      return this.ballRadiusPx(x, y) * Math.sqrt(Math.max(0, 1 - sinE * sinE));
    }

    /** Top-down position of a ball from its visible centre (via its contact point on the cloth). */
    ballToTopDown(p) { return this.toTopDown([p[0], p[1] + this.ballDropPx(p[0], p[1])]); }

    /** Screen position of a ball centre at a top-down spot (the cloth point lifted by one radius). */
    ballToCamera(td) { const c = this.toCamera(td); return [c[0], c[1] - this.ballDropPx(c[0], c[1])]; }

    /** Expected on-screen ball radius at a camera-view position. */
    ballRadiusPx(x, y) {
      const c = this.toTopDown([x, y]);
      const a = this.toCamera([c[0] - this.ballRadius, c[1]]);
      const b = this.toCamera([c[0] + this.ballRadius, c[1]]);
      return norm(sub(a, b)) / 2;
    }

    /** Where a ball rolling from start along direction first touches a cushion (top-down). */
    rayToCushion(start, direction) {
      const m = this.CUSHION * this.scale + this.ballRadius;
      const lo = [m, m], hi = [this.targetW - m, this.targetH - m];
      let tMax = Infinity;
      for (const axis of [0, 1]) {
        if (direction[axis] > 0) tMax = Math.min(tMax, (hi[axis] - start[axis]) / direction[axis]);
        else if (direction[axis] < 0) tMax = Math.min(tMax, (lo[axis] - start[axis]) / direction[axis]);
      }
      if (!isFinite(tMax)) return start;
      return add(start, mul(direction, Math.max(tMax, 0)));
    }

    /** Indices of balls (top-down) a ball rolling start->end would hit: within one diameter of its path. */
    blockers(start, end, balls) {
      const seg = sub(end, start), len2 = dot(seg, seg);
      const hits = [];
      balls.forEach((b, i) => {
        const t = len2 === 0 ? 0 : Math.min(1, Math.max(0, dot(sub(b, start), seg) / len2));
        if (norm(sub(add(start, mul(seg, t)), b)) < 2 * this.ballRadius) hits.push(i);
      });
      return hits;
    }

    /**
     * Top-down centre of the opening a ball is potted into, past the cushion jaw.
     * The nose corner is still on the cloth, so a line drawn there stops on the cushion.
     */
    pocketMouth(index) {
      const jaw = this.pockets[index], w = this.targetW, h = this.targetH;
      let outward, mm;
      if (index < 4) {
        outward = [jaw[0] < w / 2 ? -1 : 1, jaw[1] < h / 2 ? -1 : 1];
        mm = SnookerGeometry.CORNER_MOUTH_MM;
      } else if (this.portrait) {
        outward = [jaw[0] < w / 2 ? -1 : 1, 0];
        mm = SnookerGeometry.MIDDLE_MOUTH_MM;
      } else {
        outward = [0, jaw[1] < h / 2 ? -1 : 1];
        mm = SnookerGeometry.MIDDLE_MOUTH_MM;
      }
      const step = mul(outward, mm * this.scale / norm(outward));
      const mouth = add(jaw, step);
      // Stay inside the outer cushion edge. Past it, the table mapping is an
      // extrapolation and the pocket jumps away from the hole.
      const margin = 8 * this.scale;
      mouth[0] = Math.min(this.targetW - margin, Math.max(margin, mouth[0]));
      mouth[1] = Math.min(this.targetH - margin, Math.max(margin, mouth[1]));
      return mouth;
    }

    /**
     * Camera-view polygon of the one-ball-wide strip swept from start to end.
     * endOnCloth keeps the far end on the bed (a pocket opening) instead of lifting it
     * to ball-centre height, which would park it on the cushion above the hole.
     */
    corridor(start, end, endOnCloth = false) {
      const d = sub(end, start), n = norm(d);
      if (n === 0) return null;
      const side = mul([-d[1], d[0]], this.ballRadius / n);
      const pts = [add(start, side), add(end, side), sub(end, side), sub(start, side)];
      return pts.map((p, i) => (endOnCloth && (i === 1 || i === 2) ? this.toCamera(p) : this.ballToCamera(p)));
    }

    /** Aim / pot lines and the stun (90 degree) cue-ball path, all top-down. */
    calculateTrajectories(cue, target, pocket) {
      const potVec = sub(pocket, target), potDist = norm(potVec);
      if (potDist === 0) return null;
      const potUnit = mul(potVec, 1 / potDist);
      const impact = sub(target, mul(potUnit, this.ballRadius * 2));   // ghost-ball position
      const coll = sub(target, impact), cu = mul(coll, 1 / norm(coll));
      let tangent = [-cu[1], cu[0]];
      if (dot(tangent, sub(impact, cue)) < 0) tangent = mul(tangent, -1);
      return { impact, cuePath: [impact, this.rayToCushion(impact, tangent)] };
    }

    /**
     * Chooses a pocket and checks both ball paths for obstructions (camera coordinates in/out).
     * Picks the easiest pot whose paths are clear; if all are blocked, the easiest with blocked=true.
     * prefer: the pocket index chosen last time, kept unless another is clearly easier, so the
     * drawing doesn't flip between two similar pockets from frame to frame.
     */
    planShot(cueCam, targetCam, ballsCam, prefer = null, maxCutDeg = 75) {
      const cue = this.ballToTopDown(cueCam), target = this.ballToTopDown(targetCam);
      const others = [];
      for (const b of ballsCam) {
        const td = this.ballToTopDown(b);
        // Skip the cue ball and the object ball themselves (and any double detection of them). Checked
        // on screen too: along the table's length the picture is so foreshortened that a few pixels of
        // detection noise is more than a ball's width on the table.
        const nearTd = Math.min(norm(sub(td, cue)), norm(sub(td, target))) <= this.ballRadius * 1.5;
        const nearPx = [cueCam, targetCam].some(c => norm(sub(b, c)) <= 1.5 * this.ballRadiusPx(c[0], c[1]));
        if (!(nearTd || nearPx)) others.push([td, b]);
      }
      const othersTd = others.map(o => o[0]);
      const plans = [];
      this.pockets.forEach((pocket, index) => {
        const rated = this.pocketDifficulty(cue, target, index, maxCutDeg);
        if (!rated) return;
        const [difficulty, cut] = rated;
        const mouth = this.pocketMouth(index);
        const res = this.calculateTrajectories(cue, target, mouth);
        if (!res) return;
        const aimHits = this.blockers(cue, res.impact, othersTd);
        const potHits = this.blockers(target, mouth, othersTd);
        const hitIds = [...new Set([...aimHits, ...potHits])].sort((a, b) => a - b);
        // The hole is in the bed. Lifting the end to ball-centre height puts it on the
        // cushion above the opening, so the pot line is drawn on the cloth, into the hole.
        const mouthCam = this.toCamera(mouth);
        plans.push({
          pocketIndex: index,
          difficulty,
          cutDeg: cut,
          pocket: mouthCam,
          aim: [cueCam, this.ballToCamera(res.impact)],
          pot: [targetCam, mouthCam],
          cuePath: res.cuePath.map(p => this.ballToCamera(p)),
          aimCorridor: this.corridor(cue, res.impact),
          potCorridor: this.corridor(target, mouth, true),
          blockers: hitIds.map(i => others[i][1]),
          blocked: hitIds.length > 0,
          aimBlocked: aimHits.length > 0,
          potBlocked: potHits.length > 0,
        });
      });
      if (!plans.length) return null;
      plans.sort((a, b) => (a.blocked - b.blocked) || (a.difficulty - b.difficulty));
      const best = plans[0];
      const kept = plans.find(p => p.pocketIndex === prefer && p.blocked === best.blocked && p.difficulty <= 1.3 * best.difficulty);
      return kept || best;
    }
  }
  // A middle pocket only takes balls arriving within this angle of straight in; steeper ones hit the jaw
  SnookerGeometry.MIDDLE_POCKET_MAX_ENTRY_DEG = 45;
  // How far past the jaw (mm) the line has to be aimed for it to enter the opening.
  SnookerGeometry.CORNER_MOUTH_MM = 20;
  SnookerGeometry.MIDDLE_MOUTH_MM = 20;

  S.SnookerGeometry = SnookerGeometry;
})(typeof self !== 'undefined' ? self : globalThis);
