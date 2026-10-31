"""Timing primitives.

* `time_call`: one call timed three ways (GPU time by CUDA events, CPU enqueue time, wall time to completion).
* `BlockTimer`: CUDA events before and after every ViT block (forward pre-hooks and hooks), resolved once.
* `HostState`: CPU load and our process's context switches during a measurement (shared machine).
* `summarize`: median / p90 / mean / min / max of a list of milliseconds.
"""
import statistics
import time

import torch


def time_call(fn, *args):
    """Run fn(*args) once on an idle GPU and return (output, gpu_ms, cpu_enqueue_ms, wall_ms).

    gpu_ms:          CUDA events recorded just before and after the call; the GPU timestamps each event when
                     it reaches it in the stream, so this is GPU time between the two points, including any
                     idle gaps in between (e.g. waiting for the CPU to launch the next kernel).
    cpu_enqueue_ms:  host time for the Python call to return. Kernel launches are asynchronous, so this is the
                     time to *issue* the work; if it is close to gpu_ms, the CPU is the bottleneck or a sync
                     inside the call made the CPU wait.
    wall_ms:         host time from the call until the GPU has finished everything (synchronize).
    """
    torch.cuda.synchronize()  # nothing else is queued: the measurement starts from an idle GPU
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0 = time.perf_counter()
    start.record()
    out = fn(*args)
    end.record()
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    return out, start.elapsed_time(end), (t1 - t0) * 1e3, (t2 - t0) * 1e3


class BlockTimer:
    """CUDA events around each module in `blocks` (e.g. vit.blocks). Use `frames()` after the run."""

    def __init__(self, blocks):
        self._hooks, self._pending = [], []
        self._current = None
        for i, blk in enumerate(blocks):
            self._hooks.append(blk.register_forward_pre_hook(self._pre(i)))
            self._hooks.append(blk.register_forward_hook(self._post(i)))

    def _pre(self, i):
        def hook(module, args):
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self._current = (i, ev)
        return hook

    def _post(self, i):
        def hook(module, args, out):
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self._pending.append((i, self._current[1], ev))
        return hook

    def new_frame(self):
        self._pending.append(None)

    def frames(self):
        """List of per-frame lists of block ms (index = block)."""
        torch.cuda.synchronize()
        out, cur = [], None
        for item in self._pending:
            if item is None:
                if cur:
                    out.append(cur)
                cur = {}
                continue
            i, s, e = item
            cur = cur if cur is not None else {}
            cur[i] = cur.get(i, 0.0) + s.elapsed_time(e)
        if cur:
            out.append(cur)
        self._pending = []
        return [[f[i] for i in sorted(f)] for f in out]

    def remove(self):
        for h in self._hooks:
            h.remove()


class GpuStateSampler:
    """Samples SM clock, temperature and power of the physical GPU with nvidia-smi once per `period_s`, in a
    background thread, while a measurement runs. The A5000 is power-capped (230 W), so its SM clock under load moves
    with power draw and temperature; these samples show the conditions every latency number was taken under."""

    def __init__(self, gpu_index, period_s=1.0):
        import threading

        self.gpu, self.period = str(gpu_index), period_s
        self.samples, self._stop = [], threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        import subprocess

        t0 = time.perf_counter()
        while not self._stop.is_set():
            out = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu,power.draw",
                                  "--format=csv,noheader,nounits", "-i", self.gpu],
                                 capture_output=True, text=True).stdout.strip().split(", ")
            try:
                self.samples.append(dict(t=time.perf_counter() - t0, sm_mhz=float(out[0]), temp_c=float(out[1]),
                                         power_w=float(out[2])))
            except (ValueError, IndexError):
                pass
            self._stop.wait(self.period)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()

    def summary(self):
        if not self.samples:
            return {}
        return {k: summarize(s[k] for s in self.samples) for k in ("sm_mhz", "temp_c", "power_w")}


class HostState:
    """CPU-side conditions during a measurement. The machine is shared: other users' jobs can occupy every core, and
    any part of our run that is CPU-bound (Python, kernel launches, waiting for a sync to return) then slows down.
      loadavg_1m        /proc/loadavg 1-minute load at start and end (compare with cpu_count)
      running           runnable threads machine-wide (/proc/loadavg field 4), sampled every `period_s`
      involuntary_csw   times the OS took the CPU away from one of our threads because another needed it
                        (getrusage ru_nivcsw, all threads of this process): the direct measure of contention we felt
      voluntary_csw     times our threads gave the CPU up themselves (blocking waits, e.g. a CUDA sync)
      cpu_s / wall_s    CPU time used by this process (user + system) and wall time of the measurement"""

    def __init__(self, period_s=1.0):
        import threading

        self.period, self.samples, self._stop = period_s, [], threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    @staticmethod
    def _loadavg():
        f = open("/proc/loadavg").read().split()
        return float(f[0]), int(f[3].split("/")[0])

    def _run(self):
        while not self._stop.is_set():
            self.samples.append(self._loadavg()[1])
            self._stop.wait(self.period)

    @staticmethod
    def _usage():
        import resource

        r = resource.getrusage(resource.RUSAGE_SELF)
        return r.ru_utime + r.ru_stime, r.ru_nvcsw, r.ru_nivcsw

    def __enter__(self):
        self._t0, self._u0, self._l0 = time.perf_counter(), self._usage(), self._loadavg()[0]
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self._t1, self._u1, self._l1 = time.perf_counter(), self._usage(), self._loadavg()[0]

    def summary(self):
        import os

        return dict(cpu_count=os.cpu_count(), loadavg_1m_start=self._l0, loadavg_1m_end=self._l1,
                    running=summarize(self.samples), wall_s=self._t1 - self._t0, cpu_s=self._u1[0] - self._u0[0],
                    voluntary_csw=self._u1[1] - self._u0[1], involuntary_csw=self._u1[2] - self._u0[2])


def summarize(ms):
    ms = list(ms)
    if not ms:
        return {}
    s = sorted(ms)
    p = lambda q: s[min(len(s) - 1, int(round(q * (len(s) - 1))))]  # noqa: E731
    return dict(n=len(s), median=statistics.median(s), mean=statistics.fmean(s), p90=p(0.9), p10=p(0.1),
                min=s[0], max=s[-1], stdev=statistics.pstdev(s))
