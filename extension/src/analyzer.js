/*
 * Per-frame pipeline for a live stream. JavaScript port of analyzer.py.
 * All coordinates are frame pixels; `now` is in seconds.
 *
 * States:
 *   locking     no full-table view learned yet
 *   other_view  camera is on a player, a close-up, or part of the table: draw nothing
 *   moving      balls are rolling (or just appeared): draw nothing until they settle
 *   settled     balls are still: plan and draw the selected shot
 */
(function (root) {
  'use strict';
  const S = root.Snooker = root.Snooker || {};

  class SnookerAnalyzer {
    constructor(detectInterval = 0.2) {
      this.detectInterval = detectInterval;
      this.vision = new S.SnookerVision();
      this.geometry = new S.SnookerGeometry();
      this.lock = new S.TableLock();
      this.tracker = new S.BallTracker();
      this.reset();
    }

    reset() {
      this.lock.reset();
      this.tracker.reset();
      this.geometry.clearHomography();
      this.clearTarget();
      this.state = 'locking';
      this._lastDetect = null;
      this._frameShape = null;
      this._view = null;
      this._carryTarget = null;  // {pos: top-down, colour} to re-find after a camera switch
    }

    clearTarget() {
      this._pocketPref = null;  // pocket chosen for the current target, kept while it stays sensible
      this.targetId = null;
      this._targetStart = null;
      this.plan = null;
      this._planKey = null;
      this.noPocket = false;
    }

    // --- user actions ----------------------------------------------------

    toggleOrientation() {
      this.geometry.setOrientation(!this.geometry.portrait, this.lock.corners);
      this._planKey = null;
    }

    /** Selects the ball nearest to a click (frame pixels). Returns the track or null. */
    selectTarget(point) {
      const track = this.tracker.nearest(point);
      const cue = this.cueBall;
      if (!track || (cue && track.id === cue.id)) return null;
      this.targetId = track.id;
      this._targetStart = track.pos.slice();
      this._planKey = null;
      this._pocketPref = null;
      return track;
    }

    get target() { return this.targetId !== null ? (this.tracker.tracks.get(this.targetId) || null) : null; }
    get cueBall() { return this.tracker.cueBall(); }

    // --- per frame ---------------------------------------------------------

    /** frame: RGB cv.Mat. */
    process(frame, now) {
      const shape = frame.rows + 'x' + frame.cols;
      if (this._frameShape !== null && shape !== this._frameShape) this.reset();
      this._frameShape = shape;

      const { ok, quad } = this.vision.detectMainView(frame);
      const isMain = this.lock.observe(ok, quad, this.vision.tableMask, this.geometry.portrait);
      if (isMain) {
        const view = this.lock.views[this.lock.active];
        if (view !== this._view) this._switchView(view);
      }
      if (!isMain) {
        this.state = this.lock.locked ? 'other_view' : 'locking';
        this.plan = null;
        this._planKey = null;
        return;
      }

      // Detect inside the locked table area, so a player leaning over doesn't reshape it
      this.vision.tableMask = this.lock.mask;
      if (this._lastDetect === null || now - this._lastDetect >= this.detectInterval) {
        this._lastDetect = now;
        this._detect(frame, now);
      }

      if (this._carryTarget && this.tracker.balls().length) this._refindTarget();

      // The selected ball was potted or knocked away: drop the selection
      let target = this.target;
      if (this.targetId !== null &&
          (!target || Math.hypot(target.pos[0] - this._targetStart[0], target.pos[1] - this._targetStart[1]) > target.r)) {
        this.clearTarget();
        target = null;
      }

      if (!this.tracker.settled(now)) {
        this.state = 'moving';
        this.plan = null;
        this._planKey = null;
        return;
      }
      this.state = 'settled';

      const cue = this.cueBall;
      if (!target || !cue) { this.plan = null; return; }
      const key = [this.tracker.version, this.targetId, cue.id, this.geometry.portrait].join('/');
      if (key !== this._planKey) {
        this._planKey = key;
        this.plan = this.geometry.planShot(cue.pos, target.pos, this.tracker.balls().map(t => t.pos), this._pocketPref);
        this.noPocket = this.plan === null;
        if (this.plan) this._pocketPref = this.plan.pocketIndex;
      }
    }

    _switchView(view) {
      // A different camera angle: ball positions on screen all change
      const target = this.target;
      if (target && this.geometry.M) this._carryTarget = { pos: this.geometry.toTopDown(target.pos), colour: target.colour };
      this._view = view;
      this.geometry.updateHomography(this.lock.corners);
      this.tracker.reset();
      this.targetId = null;
      this._targetStart = null;
      this._planKey = null;
      this._lastDetect = null;
    }

    _refindTarget() {
      const { pos, colour } = this._carryTarget;
      this._carryTarget = null;
      let best = null;
      for (const t of this.tracker.balls()) {
        if (t.colour !== colour) continue;
        const td = this.geometry.toTopDown(t.pos);
        const d = Math.hypot(td[0] - pos[0], td[1] - pos[1]);
        if (d < 2 * this.geometry.ballRadius && (!best || d < best[0])) best = [d, t];
      }
      if (best) { this.targetId = best[1].id; this._targetStart = best[1].pos.slice(); }
    }

    _detect(frame, now) {
      const geo = this.geometry;
      const dets = this.vision.findBalls(frame, geo.playingArea(), (x, y) => geo.ballRadiusPx(x, y),
                                         (x, y) => geo.atPocketMouth(x, y));
      this.tracker.update(dets, now, (x, y, r) => this.vision.isClothAt(frame, x, y, r));
    }

    /** Plain-data snapshot for the page overlay. */
    snapshot() {
      const cue = this.cueBall, target = this.target;
      return {
        state: this.state,
        views: this.lock.views.length,
        corners: this.lock.corners,
        balls: this.tracker.balls().map(t => ({ id: t.id, x: t.pos[0], y: t.pos[1], r: t.r, colour: t.colour, visible: t.visible })),
        cueId: cue ? cue.id : null,
        targetId: target ? target.id : null,
        targetColour: target ? target.colour : null,
        plan: this.state === 'settled' ? this.plan : null,
        noPocket: this.state === 'settled' && this.noPocket,
        portrait: this.geometry.portrait,
      };
    }
  }

  S.SnookerAnalyzer = SnookerAnalyzer;
})(typeof self !== 'undefined' ? self : globalThis);
