import os, glob, sys
from pathlib import Path

# Repo-relative roots. This file used to live in the gitignored directory
# remote_results/_full_reports/_build/ and hardcoded absolute paths; it is now
# tracked, so it must work from any checkout.
REPO = Path(__file__).resolve().parents[3]
POSTPROCESS = Path(__file__).resolve().parents[1]

mroot = sys.argv[1]
RES = "remote_results"
RERUN = {
    'full_standard': '10_standard', 'full_sweep_nodes': '11_sweep_size', 'full_adjlist': '12_abl_adjlist',
    'full_labels': '13_abl_labels', 'full_color': '14_abl_color', 'full_think': '15_abl_think'}
BASE = {
    'full_standard': '01_standard', 'full_sweep_nodes': '02_sweep_size', 'full_adjlist': '04_abl_adjlist',
    'full_labels': '05_abl_labels', 'full_color': '06_abl_color', 'full_think': '07_abl_think'}


def campaign_of(path):
    parts = path.replace(chr(92), '/').split('remote_results/')
    return parts[1].split('/')[0]


def job_data_dir(job):
    flat = os.path.join(RES, job)
    if os.path.isdir(flat):
        return ('FLAT', flat)
    cands = glob.glob(os.path.join(RES, '*', '_jobs', job))
    if cands:
        cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return ('FILED', cands[0])
    return ('MISSING', flat)


problems = 0
for fam in ['full_standard', 'full_sweep_nodes', 'full_adjlist', 'full_labels', 'full_color', 'full_think']:
    man = os.path.join(mroot, fam + '.txt')
    jobs = [l.strip().split('/')[-1] for l in open(man) if l.strip() and not l.startswith('#')]
    rerun_dir = RERUN[fam]
    base_dir = BASE[fam]
    print("\n===== %s  (rerun=%s, base=%s) =====" % (fam, rerun_dir, base_dir))
    for job in jobs:
        kind, path = job_data_dir(job)
        camp = campaign_of(path)
        all_cands = glob.glob(os.path.join(RES, '*', '_jobs', job))
        camps = sorted(campaign_of(c) for c in all_cands)
        rerun_exists = os.path.isdir(os.path.join(RES, rerun_dir, '_jobs', job))
        exp = rerun_dir if rerun_exists else base_dir
        ok = (kind == 'FILED' and camp == exp)
        flag = '' if ok else '  <<< UNEXPECTED (exp %s)' % exp
        if not ok:
            problems += 1
        print("  %-52s -> %-16s [%s] copies=%s%s" % (job, camp, kind, camps, flag))

print("\n### RESOLUTION PROBLEMS: %d" % problems)
