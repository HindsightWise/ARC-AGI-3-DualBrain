# SPDX-License-Identifier: MIT-0
"""Resumable D3 local-only driver: one subprocess per cell, at most three."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
os.chdir(ROOT)
OUT=Path('runs/2026-10-22_d3-seg-groups-b4.jsonl')
RAW=Path('runs/raw/d3')
GAMES=('lp85','r11l','vc33','ft09','lf52','s5i5','su15','tn36')
done=set()
if OUT.exists():
    for line in OUT.read_text().splitlines():
        r=json.loads(line); done.add((r['game'],r['seed']))
queue=[(g,s,0) for s in range(10) for g in GAMES if (g,s) not in done]
running=[]
start=time.monotonic()
while queue or running:
    avail=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))/1024**2
    # Pause work under external memory pressure; no cell is discarded or changed.
    for job in running:
        proc=job['proc']
        if proc.poll() is None and avail < 3.5 and not job['paused']:
            os.kill(proc.pid, signal.SIGSTOP); job['paused']=True
            print('MEMORY PAUSE', job['cell'],avail,flush=True)
        elif proc.poll() is None and avail >= 4.5 and job['paused']:
            os.kill(proc.pid,signal.SIGCONT); job['paused']=False
            print('MEMORY RESUME', job['cell'],avail,flush=True)
    while queue and len(running)<3 and avail>=5:
        subprocess.run(['free','-g'],check=True)
        g,s,attempt=queue.pop(0)
        part=RAW/f'{g}-s{s}-attempt{attempt}.jsonl'
        if part.exists():
            raise RuntimeError(f'unmerged part exists; inspect before resume: {part}')
        log=RAW/f'{g}-s{s}-attempt{attempt}.log'
        cmd=['.venv/bin/python','arc3/run_local.py','--games',g,'--seeds',str(s),'--max-actions','3000','--time-budget','1200','--variant','d3-seg-groups-b4','--set','CLICK_MODE=segments','--set','CLICK_ORDER=groups','--budget-buckets','4','--out',str(part)]
        with log.open('w') as f:
            proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT)
        running.append(dict(proc=proc,cell=(g,s,attempt),part=part,log=log,paused=False))
        print('START',g,s,attempt,flush=True)
        avail-=0.75 # conservative reservation until next memory reading
    for job in running[:]:
        code=job['proc'].poll()
        if code is None: continue
        g,s,attempt=job['cell']; running.remove(job)
        if code:
            if 'None frame data' in job['log'].read_text() and attempt<3:
                queue.insert(0,(g,s,attempt+1))
                print('RETRY None frame data',g,s,attempt,flush=True)
                continue
            for other in running:
                if other['paused']: os.kill(other['proc'].pid,signal.SIGCONT)
                other['proc'].terminate()
            raise RuntimeError(f'failed {job["cell"]}: {job["log"]}')
        rows=[json.loads(l) for l in job['part'].read_text().splitlines()]
        assert rows and all(r['game']==g and r['seed']==s for r in rows)
        with OUT.open('a') as f:
            for row in rows: f.write(json.dumps(row,separators=(',',':'))+'\n')
            f.flush(); os.fsync(f.fileno())
        done.add((g,s))
        print('DONE',g,s,'cells',len(done),'elapsed',round(time.monotonic()-start),'available',round(avail,2),flush=True)
    time.sleep(2)
print('COMPLETE',len(done),round(time.monotonic()-start),flush=True)
