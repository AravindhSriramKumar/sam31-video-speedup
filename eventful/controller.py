"""Finds the ViT trunk and swaps its blocks for EventfulBlocks (and back).

Nothing in SAM is edited: `vit.blocks` is a ModuleList attribute, so assigning another ModuleList is enough.
ViT.forward only iterates `self.blocks` and checks the block index against `full_attn_ids`, so the block order
and count must stay the same, which they do.
"""
import torch
import torch.nn as nn

from eventful.block import EventfulBlock


def find_vit(model):
    """The single sam3.model.vitdet.ViT inside `model` (the predictor deletes the tracker's own backbone)."""
    from sam3.model.vitdet import ViT

    found = [m for m in model.modules() if isinstance(m, ViT)]
    if len(found) != 1:
        raise RuntimeError(f"expected exactly one ViT, found {len(found)}")
    return found[0]


class EventfulController:
    """Owns the block swap and the gates' frame bookkeeping.

    One trunk call = one frame (batch 1). Drivers (benchmark, pipeline adapter) call `run(x)`, which runs the
    frame and then `step()`. With a compiled trunk attached (`attach_compiled`), flush frames run through the
    eager ViT and steady frames through the compiled one, so the compiled graph never sees a flush.
    With `cfg.refresh_every = R`, every R-th frame after a flush is computed densely again (all gates flushed).
    Call `reset()` between videos.
    """

    def __init__(self, vit, cfg=None, block_cls=EventfulBlock, stats=None):
        from eventful.config import resolve_k_schedule

        self.vit, self.cfg, self.stats = vit, cfg, stats
        self.k_schedule = resolve_k_schedule(cfg) if cfg is not None else None
        self.orig_blocks = vit.blocks
        self.ev_blocks = nn.ModuleList([block_cls(b, i, cfg, stats=stats, k_schedule=self.k_schedule)
                                        if cfg is not None else block_cls(b, i, None)
                                        for i, b in enumerate(vit.blocks)])
        self.frames_since_flush = 0
        self.frames = 0
        self.compiled = None
        # lagged K (cfg.lag_tau): per-frame K for all gates from the previous frame's counts
        self.lagged = cfg is not None and cfg.lag_tau is not None
        self.k0 = max(self.k_schedule["mlp"]) if self.k_schedule is not None else None
        self.k_hist = []                                  # K used by every steady frame (for reporting reuse)
        self._host = self._ev = None
        self._pending, self._late, self.late_waits = False, 0, 0
        self.fallback_k = cfg.fallback_k if cfg is not None else None
        self.fallbacks = 0                                # frames run as a dense flush by the fallback
        # The first steady frame after a flush measures one frame of change against freshly filled references, so its
        # count is far too low (K would collapse to the smallest bucket on the next frame). Its count is not used:
        # K stays K0 for one more frame. Only for the configurations that opt in (dense fallback), so earlier
        # runs stay reproducible.
        self._skip_count = False
        self.set_enabled(True)

    # ------------------------------------------------------------------ swap
    def set_enabled(self, on: bool):
        self.vit.blocks = self.ev_blocks if on else self.orig_blocks

    @property
    def enabled(self):
        return self.vit.blocks is self.ev_blocks

    def restore(self):
        self.set_enabled(False)

    # ------------------------------------------------------------------ state
    def gates(self):
        """Yield (block_idx, name, gate) for every installed gate."""
        for i, b in enumerate(self.ev_blocks):
            for name in ("qkv_gate", "proj_gate", "mlp_gate"):
                g = getattr(b, name)
                if g is not None:
                    yield i, name, g

    def reset(self):
        """Forget all cached state: the next frame is computed densely (a "flush")."""
        for _, _, g in self.gates():
            g.reset()
        self.frames_since_flush = 0
        if self.lagged:
            self._pending, self._late = False, 0
            self.set_k(self.k0)

    # ------------------------------------------------------------------ lagged K
    def bucket(self, count):
        b, n = self.cfg.k_bucket, max(g.ref.shape[0] if g.ref is not None else 0 for _, _, g in self.gates()) or 5184
        return int(min(n, max(b, b * -(-int(count) // b))))

    def set_k(self, k):
        for _, _, g in self.gates():
            g.k = int(k)

    def buckets(self, n=5184):
        b = self.cfg.k_bucket
        return sorted({min(n, b * i) for i in range(1, -(-n // b) + 1)})

    def _read_counts(self):
        """Start of a frame: if last frame's counts have reached the host, set K from them. If not, keep the current
        K for this frame (a two-frame lag); if they are already a frame late, wait for them, so K is never more than
        two frames old. (With SAM's own per-step synchronisations the counts are normally ready: no wait.)"""
        if not self._pending:
            return
        if not self._ev.query():
            self._late += 1
            if self._late < 2:
                return
            self._ev.synchronize()
            self.late_waits += 1
        self.set_k(self.bucket(float(self._host.float().mean())))
        self._pending, self._late = False, 0

    def _send_counts(self):
        """End of a steady frame: copy every gate's count to pinned host memory without waiting."""
        cs = [g.count for _, _, g in self.gates() if g.count is not None]
        if not cs:
            return
        if self._host is None:
            self._host = torch.empty(len(cs), dtype=torch.int64, pin_memory=True)
            self._ev = torch.cuda.Event()
        self._host.copy_(torch.stack(cs), non_blocking=True)
        self._ev.record()
        if not self._pending:
            self._late = 0
        self._pending = True

    def warm_buckets(self, x, frames=3):
        """Compile the steady graph for every K bucket before anything is timed (one graph per bucket)."""
        import torch._dynamo.config as dcfg
        dcfg.recompile_limit = max(dcfg.recompile_limit, 64)
        dcfg.accumulated_recompile_limit = max(getattr(dcfg, "accumulated_recompile_limit", 256), 512)
        for k in self.buckets():
            self.reset()
            for _ in range(frames):
                self.set_k(k)
                self._pending = False
                self.run(x)
        if self.fallback_k is not None:                   # the compiled flush graphs (video starts, fallbacks)
            self.reset()
            self.run(x)
            self.run(x)
            self.set_k(self.fallback_k)
            self._pending = False
            self.run(x)                                    # a fallback frame (counting flush)
        self.reset()
        self.k_hist = []
        self.fallbacks = 0

    def take_k_hist(self):
        h, self.k_hist = self.k_hist, []
        return h

    # ------------------------------------------------------------------ running frames
    def attach_compiled(self, compiled_trunk):
        """A torch.compile'd version of self.vit, used for steady (non-flush) frames."""
        self.compiled = compiled_trunk

    def needs_flush(self):
        return any(g.needs_flush for _, _, g in self.gates())

    def run(self, x):
        """One frame: eager ViT if any gate must flush (or nothing is compiled), compiled ViT otherwise. With a dense
        fallback configured, flush frames also run compiled (a separate compiled graph), and a frame whose lagged K
        reaches fallback_k becomes a flush: dense compute that refreshes every cache; the next frame restarts at K0."""
        flush = self.needs_flush()
        fell_back = False
        if self.lagged and not flush:
            self._read_counts()
            if self.fallback_k is not None and next(g.k for _, _, g in self.gates()) >= self.fallback_k:
                for _, _, g in self.gates():
                    g.reset()
                    g.count_on_flush = True
                flush = fell_back = True
                self.fallbacks += 1
        # The very first flush allocates the gate caches: always eager, so the caches are ordinary allocations (never
        # memory owned by a compiled graph / CUDA graph, which a later replay could overwrite).
        allocated = all(g.ref is not None for _, _, g in self.gates())
        compiled_ok = self.compiled is not None and (not flush or (self.fallback_k is not None and allocated))
        trunk = self.compiled if compiled_ok else self.vit
        if compiled_ok:
            # each frame is a new step for CUDA graphs (mode "reduce-overhead"): outputs of the previous replay may be
            # overwritten; callers consume the outputs before the next frame (the adapter concatenates them)
            torch.compiler.cudagraph_mark_step_begin()
        out = trunk(x)
        if self.lagged and not flush:
            self.k_hist.append(next(g.k for _, _, g in self.gates()))
            if self._skip_count:
                self._skip_count = False
            else:
                self._send_counts()
        elif self.lagged and flush and fell_back:
            # a fallback frame: dense, but its counts (measured against the old ref) set the next frame's K
            self.k_hist.append(max(g.out.shape[0] for _, _, g in self.gates()))   # a dense frame: every token
            for _, _, g in self.gates():
                g.count_on_flush = False
            self._send_counts()
        elif self.lagged and flush:                       # reset (video start, out of order): restart at K0
            self._pending, self._late = False, 0
            self.set_k(self.k0)
            self._skip_count = self.fallback_k is not None
        self.step()
        return out

    def step(self):
        """After every frame: count frames, schedule a refresh flush."""
        self.frames += 1
        self.frames_since_flush += 1
        R = self.cfg.refresh_every if self.cfg is not None else None
        if R and self.frames_since_flush >= R:
            self.reset()

    def cache_bytes(self):
        """Bytes held by all gate caches (ref + out), once allocated."""
        return sum(g.cache_bytes() for _, _, g in self.gates())
