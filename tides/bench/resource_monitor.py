"""Resource monitor — samples CPU, RAM, GPU, VRAM during benchmark runs.

Uses pynvml for GPU metrics (utilization, memory, temperature, power, clocks)
and psutil for CPU/RAM. Provides per-run detailed resource profiles.

Usage:
    monitor = ResourceMonitor(interval=0.5)
    monitor.start()
    ... run calculation ...
    stats = monitor.stop()  # returns dict with peak/avg resource usage

Metrics collected per sample:
    CPU: proc_cpu_pct, sys_cpu_pct, proc_threads, proc_rss_mb,
         sys_ram_used_mb, sys_ram_available_mb
    GPU: gpu_util_pct, gpu_mem_util_pct, vram_used_mb, vram_total_mb,
         vram_free_mb, gpu_temp_c, gpu_power_w, gpu_clock_mhz, gpu_mem_clock_mhz

Summary stats per run:
    For each metric: _avg, _peak, _min (where applicable)
    Plus: n_samples, duration_s, cpu_cores, ram_total_mb, gpu_name
"""
import time, threading, subprocess, os, sys, json

_PSUTIL_AVAILABLE = False
try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    pass

_PYNVML_AVAILABLE = False
_NVML_INITIALIZED = False
try:
    import pynvml
    _PYNVML_AVAILABLE = True
except ImportError:
    pass


def _init_nvml():
    global _NVML_INITIALIZED
    if _PYNVML_AVAILABLE and not _NVML_INITIALIZED:
        pynvml.nvmlInit()
        _NVML_INITIALIZED = True


def _shutdown_nvml():
    global _NVML_INITIALIZED
    if _PYNVML_AVAILABLE and _NVML_INITIALIZED:
        pynvml.nvmlShutdown()
        _NVML_INITIALIZED = False


def _query_gpu_pynvml(handle):
    """Query GPU via pynvml. Returns dict with detailed GPU metrics."""
    try:
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW -> W
        clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
        mem_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
        return {
            'gpu_util_pct': float(util.gpu),
            'gpu_mem_util_pct': float(util.memory),
            'vram_used_mb': float(mem.used) / (1024 * 1024),
            'vram_total_mb': float(mem.total) / (1024 * 1024),
            'vram_free_mb': float(mem.free) / (1024 * 1024),
            'gpu_temp_c': float(temp),
            'gpu_power_w': float(power),
            'gpu_clock_mhz': float(clock),
            'gpu_mem_clock_mhz': float(mem_clock),
        }
    except Exception:
        return None


def _query_gpu_nvidia_smi():
    """Fallback: query nvidia-smi subprocess."""
    r = subprocess.run(
        ['nvidia-smi',
         '--query-gpu=utilization.gpu,memory.used,memory.total,memory.free,temperature.gpu,power.draw,clocks.sm,clocks.mem',
         '--format=csv,noheader,nounits'],
        capture_output=True, text=True, timeout=5)
    if r.returncode != 0 or not r.stdout.strip():
        return None
    parts = r.stdout.strip().split(', ')
    if len(parts) < 8:
        return None
    return {
        'gpu_util_pct': float(parts[0]),
        'vram_used_mb': float(parts[1]),
        'vram_total_mb': float(parts[2]),
        'vram_free_mb': float(parts[3]),
        'gpu_temp_c': float(parts[4]),
        'gpu_power_w': float(parts[5]),
        'gpu_clock_mhz': float(parts[6]),
        'gpu_mem_clock_mhz': float(parts[7]),
        'gpu_mem_util_pct': 0.0,
    }


def get_gpu_handle():
    """Get GPU handle for pynvml, or None if unavailable."""
    if not _PYNVML_AVAILABLE:
        return None
    _init_nvml()
    try:
        return pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception:
        return None


def get_system_info():
    """Collect static system info (called once at script start)."""
    info = {
        'cpu_count': 0,
        'cpu_count_physical': 0,
        'cpu_model': 'unknown',
        'ram_total_mb': 0,
        'gpu_name': 'unknown',
        'gpu_vram_total_mb': 0,
        'gpu_driver_version': 'unknown',
        'gpu_cuda_version': 'unknown',
        'psutil_available': _PSUTIL_AVAILABLE,
        'pynvml_available': _PYNVML_AVAILABLE,
    }
    if _PSUTIL_AVAILABLE:
        info['cpu_count'] = psutil.cpu_count(logical=True)
        info['cpu_count_physical'] = psutil.cpu_count(logical=False)
        info['ram_total_mb'] = round(psutil.virtual_memory().total / (1024 * 1024), 2)
        freq = psutil.cpu_freq()
        if freq:
            info['cpu_freq_max_mhz'] = freq.max
        # CPU model
        try:
            with open('/proc/cpuinfo') as f:
                for line in f:
                    if 'model name' in line:
                        info['cpu_model'] = line.split(':')[1].strip()
                        break
        except Exception:
            pass
    # GPU info via pynvml
    if _PYNVML_AVAILABLE:
        _init_nvml()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info['gpu_name'] = pynvml.nvmlDeviceGetName(handle)
            if isinstance(info['gpu_name'], bytes):
                info['gpu_name'] = info['gpu_name'].decode()
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            info['gpu_vram_total_mb'] = round(mem.total / (1024 * 1024), 2)
            info['gpu_driver_version'] = pynvml.nvmlSystemGetDriverVersion()
            if isinstance(info['gpu_driver_version'], bytes):
                info['gpu_driver_version'] = info['gpu_driver_version'].decode()
            info['gpu_cuda_version'] = pynvml.nvmlSystemGetCudaDriverVersion()
        except Exception:
            pass
    else:
        # Fallback: nvidia-smi
        r = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5)
        if r.returncode == 0 and r.stdout.strip():
            parts = r.stdout.strip().split(', ')
            info['gpu_name'] = parts[0].strip()
            if len(parts) >= 2:
                info['gpu_vram_total_mb'] = float(parts[1].strip().replace(' MiB', ''))
            if len(parts) >= 3:
                info['gpu_driver_version'] = parts[2].strip()
    return info


# Subprocess sampler script — runs independently so it works even when
# native C++ code holds the GIL in the parent process.
_SAMPLER_SCRIPT = '''
import sys, os, time, json
pid = int(sys.argv[1])
interval = float(sys.argv[2])
out_path = sys.argv[3]
stop_file = out_path + ".stop"

import psutil
try:
    import pynvml
    pynvml.nvmlInit()
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    gpu_handle = None

proc = psutil.Process(pid)
samples = []
t_start = time.time()
# Prime cpu_percent
proc.cpu_percent()
psutil.cpu_percent()

while True:
    t = time.time()
    sample = {"t": t}
    # CPU/RAM
    sample["proc_cpu_pct"] = proc.cpu_percent()
    mem = proc.memory_info()
    sample["proc_rss_mb"] = mem.rss / (1024 * 1024)
    sample["proc_threads"] = proc.num_threads()
    vm = psutil.virtual_memory()
    sample["sys_ram_used_mb"] = (vm.total - vm.available) / (1024 * 1024)
    sample["sys_ram_available_mb"] = vm.available / (1024 * 1024)
    sample["sys_ram_pct"] = vm.percent
    sample["sys_cpu_pct"] = psutil.cpu_percent()
    # GPU
    if gpu_handle is not None:
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
            temp = pynvml.nvmlDeviceGetTemperature(gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
            power = pynvml.nvmlDeviceGetPowerUsage(gpu_handle) / 1000.0
            clock = pynvml.nvmlDeviceGetClockInfo(gpu_handle, pynvml.NVML_CLOCK_SM)
            mem_clock = pynvml.nvmlDeviceGetClockInfo(gpu_handle, pynvml.NVML_CLOCK_MEM)
            sample["gpu_util_pct"] = float(util.gpu)
            sample["gpu_mem_util_pct"] = float(util.memory)
            sample["vram_used_mb"] = float(mem_info.used) / (1024 * 1024)
            sample["vram_total_mb"] = float(mem_info.total) / (1024 * 1024)
            sample["vram_free_mb"] = float(mem_info.free) / (1024 * 1024)
            sample["gpu_temp_c"] = float(temp)
            sample["gpu_power_w"] = float(power)
            sample["gpu_clock_mhz"] = float(clock)
            sample["gpu_mem_clock_mhz"] = float(mem_clock)
        except Exception:
            pass
    samples.append(sample)
    time.sleep(interval)
    # Check stop signal
    if os.path.exists(stop_file):
        break
    if not proc.is_running() or time.time() - t_start > 3600:
        break

with open(out_path, "w") as f:
    json.dump(samples, f)
'''


class ResourceMonitor:
    """Subprocess-based resource monitor that works even when native C++ holds the GIL.

    Spawns a child process that samples CPU/RAM/GPU/VRAM at fixed intervals
    and writes results to a temp file. This is necessary because TIDES native
    code blocks the Python GIL, preventing daemon threads from running.
    """

    def __init__(self, interval=0.5, pid=None):
        self.interval = interval
        self.pid = pid or os.getpid()
        self._proc = None
        self._tmpfile = None

    def start(self):
        import tempfile
        self._tmpfile = tempfile.mktemp(suffix='.json', prefix='resource_monitor_')
        script_path = tempfile.mktemp(suffix='.py', prefix='resource_sampler_')
        with open(script_path, 'w') as f:
            f.write(_SAMPLER_SCRIPT)
        self._proc = subprocess.Popen(
            [sys.executable, script_path, str(self.pid), str(self.interval), self._tmpfile],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def stop(self):
        if self._proc is None:
            return {'error': 'monitor not started'}
        # Create stop file to signal the sampler to exit and write output
        stop_file = self._tmpfile + '.stop'
        with open(stop_file, 'w') as f:
            f.write('stop')
        # Wait for sampler to finish writing
        self._proc.wait(timeout=15)
        time.sleep(0.1)
        samples = []
        if self._tmpfile and os.path.exists(self._tmpfile):
            with open(self._tmpfile) as f:
                samples = json.load(f)
            os.unlink(self._tmpfile)
        if os.path.exists(stop_file):
            os.unlink(stop_file)
        self._samples = samples
        return self._summarize()

    def _summarize(self):
        if not self._samples:
            return {'error': 'no samples collected'}
        s = self._samples
        n = len(s)
        result = {
            'n_samples': n,
            'interval_s': self.interval,
            'duration_s': round(s[-1]['t'] - s[0]['t'], 2) if n > 1 else 0,
        }
        # All numeric metrics to summarize
        metrics = [
            'proc_cpu_pct', 'sys_cpu_pct', 'proc_threads',
            'proc_rss_mb', 'sys_ram_used_mb', 'sys_ram_available_mb', 'sys_ram_pct',
            'gpu_util_pct', 'gpu_mem_util_pct',
            'vram_used_mb', 'vram_free_mb',
            'gpu_temp_c', 'gpu_power_w',
            'gpu_clock_mhz', 'gpu_mem_clock_mhz',
        ]
        for key in metrics:
            vals = [x[key] for x in s if key in x]
            if not vals:
                continue
            result[f'{key}_avg'] = round(sum(vals) / len(vals), 2)
            result[f'{key}_peak'] = round(max(vals), 2)
            result[f'{key}_min'] = round(min(vals), 2)
        # VRAM total (constant)
        vram_totals = [x['vram_total_mb'] for x in s if 'vram_total_mb' in x]
        if vram_totals:
            result['vram_total_mb'] = round(vram_totals[0], 2)
        # System constants
        if _PSUTIL_AVAILABLE:
            result['cpu_cores'] = psutil.cpu_count(logical=True)
            result['cpu_cores_physical'] = psutil.cpu_count(logical=False)
            result['ram_total_mb'] = round(psutil.virtual_memory().total / (1024 * 1024), 2)
        return result
