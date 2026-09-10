#!/usr/bin/env python3
"""Record isolated test-container resource use, including both torchrun ranks."""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time


def read_int(path):
    try:return int(Path(path).read_text().strip())
    except (OSError,ValueError):return None


def cpu_usec():
    try:
        values=dict(line.split() for line in Path('/sys/fs/cgroup/cpu.stat').read_text().splitlines())
        return int(values['usage_usec'])
    except (OSError,KeyError,ValueError):return None


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--timeout',type=int,default=3600)
    p.add_argument('--no-gpu',action='store_true')
    p.add_argument('--max-memory-gib',type=int)
    p.add_argument('--min-host-memory-gib',type=int)
    p.add_argument('command',nargs=argparse.REMAINDER)
    args=p.parse_args();command=args.command
    if command and command[0]=='--':command=command[1:]
    if not command or args.timeout<=0:p.error('positive timeout and command required')
    args.output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();initial_cpu=cpu_usec();stop=threading.Event();samples=[];errors=[];abort_reasons=[];process_holder=[];abort_started=[None]
    def collect():
        while not stop.is_set():
            row={'elapsed_s':time.monotonic()-start,'memory_current_bytes':read_int('/sys/fs/cgroup/memory.current')}
            try:
                row['memory_stat']={k:int(v) for k,v in (line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines()) if k in {'anon','file','kernel','inactive_file','active_file','file_mapped'}}
            except (OSError,ValueError):row['memory_stat']=None
            available=next((int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')),0)
            row['host_memory_available_bytes']=available
            reason=None
            if args.max_memory_gib and row['memory_current_bytes'] is None:reason='memory_telemetry_unavailable'
            elif args.max_memory_gib and row['memory_current_bytes']>args.max_memory_gib*2**30:reason='container_memory_guard'
            if args.min_host_memory_gib and available<args.min_host_memory_gib*2**30:reason='host_memory_guard'
            if reason and process_holder and process_holder[0].poll() is None:
                abort_reasons.append(reason)
                if abort_started[0] is None:abort_started[0]=time.monotonic()
                sig=signal.SIGKILL if time.monotonic()-abort_started[0]>3 else signal.SIGTERM
                try:os.killpg(process_holder[0].pid,sig)
                except ProcessLookupError:pass
            try:
                raw='' if args.no_gpu else subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,utilization.gpu,power.draw','--format=csv,noheader,nounits'],text=True,timeout=8)
                row['gpus']=[{'uuid':v[0].strip(),'memory_mib':float(v[1]),'utilization_pct':float(v[2]),'power_w':float(v[3])} for line in raw.splitlines() if (v:=line.split(','))]
            except Exception as e:errors.append(type(e).__name__)
            samples.append(row)
            with (args.output/'resources.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            stop.wait(1)
    monitor=threading.Thread(target=collect,daemon=True);monitor.start()
    timed_out=False
    with (args.output/'process.log').open('w') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        process_holder.append(child)
        try:code=child.wait(timeout=args.timeout)
        except (subprocess.TimeoutExpired,KeyboardInterrupt):
            timed_out=True;os.killpg(child.pid,signal.SIGTERM)
            try:code=child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait()
        finally:stop.set();monitor.join(timeout=10)
    elapsed=time.monotonic()-start;final_cpu=cpu_usec()
    uuids={g['uuid'] for s in samples for g in s.get('gpus',[])}
    summaries={}
    for uuid in uuids:
        observed=[(s['elapsed_s'],g) for s in samples for g in s.get('gpus',[]) if g['uuid']==uuid]
        joules=sum((b[0]-a[0])*(a[1]['power_w']+b[1]['power_w'])/2 for a,b in zip(observed,observed[1:]))
        summaries[uuid]={'sampled_peak_memory_mib':max(g['memory_mib'] for _,g in observed),
                         'sampled_mean_utilization_pct':sum(g['utilization_pct'] for _,g in observed)/len(observed),
                         'sampled_energy_wh':joules/3600}
    result={'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':command,
            'exit_code':code,'timed_out':timed_out,'memory_guard_aborts':abort_reasons,'process_wall_s':elapsed,
            'visible_gpu_minutes':len(uuids)*elapsed/60,
            'container_cpu_core_seconds':(final_cpu-initial_cpu)/1e6 if final_cpu is not None and initial_cpu is not None else None,
            'container_lifetime_memory_peak_bytes':read_int('/sys/fs/cgroup/memory.peak'),
            'sampled_memory_peak_bytes':max((s['memory_current_bytes'] for s in samples if s['memory_current_bytes'] is not None),default=None),
            'gpus':summaries,'monitor_errors':errors,
            'measurement_notes':'GPU energy is sampled board power, not facility billing; GPU minutes exclude image pulling and scheduling; cgroup memory includes file cache and container initialization, not only model tensors.'}
    (args.output/'resources-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    return code if 0<=code<=255 else 1


if __name__=='__main__':raise SystemExit(main())
