/*
 * Ball tracking across frames of a live stream. JavaScript port of tracking.py.
 * - Balls keep a stable id and a colour decided by majority vote over recent frames; a
 *   detection only continues a track of a compatible colour (a white is never a red).
 * - A vanished ball is only forgotten when bare cloth shows where it was (potted/moved);
 *   if something covers it (a player leaning over), its last position is kept.
 * - "Settled" means nothing moved for settleTime seconds; lines are only drawn then.
 */
(function (root) {
  'use strict';
  const S = root.Snooker = root.Snooker || {};

  // Colours that one ball can plausibly flicker between from frame to frame
  const FAMILY = { red: 'reddish', pink: 'reddish', brown: 'reddish' };
  /** Can a track labelled a be the same ball as a detection labelled b? */
  function compatible(a, b) {
    if (a === 'unknown' || b === 'unknown') return true;
    return (FAMILY[a] || a) === (FAMILY[b] || b);
  }

  class Track {
    constructor(id, pos, r, colour, now) {
      this.id = id;
      this.pos = pos.slice();
      this.shown = pos.slice();   // position the last shot plan was computed from
      this.suspect = false;       // last detection was a jump: move if it happens again
      this.r = r;
      this.votes = new Map([[colour, 1]]);
      this.recent = [colour];   // latest labels (up to 15); the colour follows these
      this.hits = 1;
      this.lastSeen = now;
      this.visible = true;
    }
    get colour() {
      // Majority of the recent labels, ignoring "unknown" unless that's all there is
      const counts = new Map();
      for (const c of this.recent) counts.set(c, (counts.get(c) || 0) + 1);
      const sorted = [...counts.entries()].sort((a, b) => b[1] - a[1] || this.recent.indexOf(a[0]) - this.recent.indexOf(b[0]));
      for (const [c] of sorted) if (c !== 'unknown') return c;
      return 'unknown';
    }
    get confirmed() { return this.hits >= 2; }
    vote(colour) {
      this.votes.set(colour, (this.votes.get(colour) || 0) + 1);
      this.recent.push(colour);
      if (this.recent.length > 15) this.recent.shift();
    }
    recentCount(colour) { return this.recent.filter(c => c === colour).length; }
  }

  class BallTracker {
    // moveTol: movement (in ball radii) between detections that counts as moving. Smaller shifts are
    // detection noise on a still ball and are smoothed out instead of making the lines jump.
    constructor(moveTol = 0.6, settleTime = 0.6, matchRadius = 1.5) {
      this.moveTol = moveTol;
      this.settleTime = settleTime;
      this.matchRadius = matchRadius;
      this.reset();
    }

    reset() {
      this.tracks = new Map();
      this.nextId = 0;
      this.lastMotion = null;
      this.version = 0;
    }

    /** detections: [x, y, r, colour]; isClothAt(x, y, r) -> bare cloth visible there. */
    update(detections, now, isClothAt) {
      let moved = false, changed = false;
      const freeDets = new Set(detections.map((_, i) => i));
      const freeTracks = new Set(this.tracks.keys());

      // Greedy nearest-first matching. A detection right on top of a track is the same ball even if
      // its colour label flickered (a warm-lit white read as brown for a frame, say).
      let pairs = [];
      for (const [tid, t] of this.tracks) {
        detections.forEach(([x, y, r, colour], di) => {
          const d = Math.hypot(x - t.pos[0], y - t.pos[1]);
          const sameSpot = d < this.moveTol * Math.max(r, t.r);
          if (d < this.matchRadius * Math.max(r, t.r) && (compatible(t.colour, colour) || sameSpot)) pairs.push([d, tid, di]);
        });
      }
      pairs.sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2]);
      for (const [d, tid, di] of pairs) {
        if (!freeTracks.has(tid) || !freeDets.has(di)) continue;
        const t = this.tracks.get(tid);
        const [x, y, r, colour] = detections[di];
        // A real move shows up in two detections running (a struck ball keeps going); a one-off
        // jump is a detection glitch
        if (d > this.moveTol * r && t.suspect) {
          if (this._move(t, x, y)) moved = true;
          changed = true;
        } else if (d > this.moveTol * r) {
          t.suspect = true;
        } else {
          t.suspect = false;
          // Jitter on a still ball: average it out; replan only once it adds up
          t.pos = [t.pos[0] + 0.2 * (x - t.pos[0]), t.pos[1] + 0.2 * (y - t.pos[1])];
          if (Math.hypot(t.pos[0] - t.shown[0], t.pos[1] - t.shown[1]) > 0.2 * r) { t.shown = t.pos.slice(); changed = true; }
        }
        if (this._seen(t, r, colour, now)) changed = true;
        freeTracks.delete(tid);
        freeDets.delete(di);
      }

      // Second pass: a ball that vanished from its spot while a new one appeared close by has simply
      // rolled there (too far for the first pass, e.g. the last slow roll before stopping). Without
      // this the old spot would be kept as a "covered" ball: a phantom next to the real one.
      pairs = [];
      for (const tid of freeTracks) {
        const t = this.tracks.get(tid);
        for (const di of freeDets) {
          const [x, y, r, colour] = detections[di];
          const d = Math.hypot(x - t.pos[0], y - t.pos[1]);
          if (d < 3 * Math.max(r, t.r) && compatible(t.colour, colour)) pairs.push([d, tid, di]);
        }
      }
      pairs.sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2]);
      for (const [, tid, di] of pairs) {
        if (!freeTracks.has(tid) || !freeDets.has(di)) continue;
        const t = this.tracks.get(tid);
        const [x, y, r, colour] = detections[di];
        if (this._move(t, x, y)) moved = true;
        this._seen(t, r, colour, now);
        changed = true;
        freeTracks.delete(tid);
        freeDets.delete(di);
      }

      // Not seen this time: gone if the cloth shows through, otherwise covered
      for (const tid of freeTracks) {
        const t = this.tracks.get(tid);
        if (isClothAt(t.pos[0], t.pos[1], t.r)) {
          if (t.confirmed) moved = true;
          this.tracks.delete(tid);
          changed = true;
        } else {
          t.visible = false;
        }
      }

      for (const di of freeDets) {
        const [x, y, r, colour] = detections[di];
        this.tracks.set(this.nextId, new Track(this.nextId, [x, y], r, colour, now));
        this.nextId++;
        changed = true;
      }

      // A new ball confirmed this round means something rolled into view
      for (const t of this.tracks.values()) {
        if (t.hits === 2 && t.lastSeen === now && this.lastMotion !== null) moved = true;
      }

      if (moved || this.lastMotion === null) this.lastMotion = now;
      if (changed) this.version++;
    }

    /** Moves a track to a new spot; returns true if that counts as a ball moving. */
    _move(t, x, y) {
      t.pos = [x, y];
      t.shown = [x, y];
      t.suspect = false;
      return t.confirmed;
    }

    /** Records a sighting; returns true if the track had been hidden (so the drawing changes). */
    _seen(t, r, colour, now) {
      t.r = r;
      t.vote(colour);
      t.hits++;
      t.lastSeen = now;
      const wasHidden = !t.visible;
      t.visible = true;
      return wasHidden;
    }

    settled(now) { return this.lastMotion !== null && now - this.lastMotion >= this.settleTime && this.tracks.size > 0; }

    balls() { return [...this.tracks.values()].filter(t => t.confirmed); }

    cueBall() {
      const whites = this.balls().filter(t => t.colour === 'white');
      if (!whites.length) return null;
      // Prefer the one actually in view, then the one most consistently seen as white lately
      const key = t => [t.visible ? 1 : 0, t.recentCount('white')];
      return whites.reduce((a, b) => { const ka = key(a), kb = key(b); return (kb[0] - ka[0] || kb[1] - ka[1]) > 0 ? b : a; });
    }

    nearest(pos, maxRadii = 2.0) {
      let best = null;
      for (const t of this.balls()) {
        const d = Math.hypot(pos[0] - t.pos[0], pos[1] - t.pos[1]);
        if (d < maxRadii * t.r && (!best || d < best[0])) best = [d, t];
      }
      return best ? best[1] : null;
    }
  }

  S.BallTracker = BallTracker;
})(typeof self !== 'undefined' ? self : globalThis);
