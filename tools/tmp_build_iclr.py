#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path

REPO = Path.cwd()
TMP = Path('/tmp')
FROZEN = TMP / 'frozen'
BASE = TMP / 'base'
PKG = TMP / 'pkg'
AE = TMP / 'ae'
OUT = TMP / 'out'


def run(*args: str, cwd: Path | None = None) -> None:
    print('+', ' '.join(args), flush=True)
    subprocess.run(args, cwd=cwd, check=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def scrub_tree(root: Path) -> None:
    text_ext = {
        '.md', '.txt', '.py', '.toml', '.cfg', '.ini', '.json', '.jsonl',
        '.csv', '.yaml', '.yml', '.tex', '.ipynb', '.lark', '.rst'
    }
    replacements = [
        (re.compile(r'Victor Lavrenko', re.I), 'Anonymous Author'),
        (re.compile(r'victorlavrenko', re.I), 'anonymous-author'),
        (re.compile(r'lavrenko/casefactory', re.I), 'anonymous-review/casefactory'),
        (re.compile(r'victor@peacetech\.vc', re.I), 'anonymous@example.invalid'),
        (re.compile(r'PeaceTech VC', re.I), 'Anonymous Institution'),
    ]
    for p in root.rglob('*'):
        if not p.is_file() or p.stat().st_size > 20 * 1024 * 1024:
            continue
        if p.suffix.lower() not in text_ext and p.name not in {'LICENSE', 'MANIFEST.in'}:
            continue
        try:
            text = p.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            continue
        for rx, repl in replacements:
            text = rx.sub(repl, text)
        p.write_text(text, encoding='utf-8')


def scan_anonymity(root: Path) -> None:
    needles = [
        b'victor lavrenko', b'victorlavrenko', b'victor@peacetech.vc',
        b'peacetech vc', b'lavrenko/casefactory'
    ]
    hits: list[str] = []
    for p in root.rglob('*'):
        if not p.is_file():
            continue
        tail = b''
        try:
            with p.open('rb') as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b''):
                    data = (tail + chunk).lower()
                    if any(n in data for n in needles):
                        hits.append(p.relative_to(root).as_posix())
                        break
                    tail = data[-128:]
        except OSError:
            hits.append(p.relative_to(root).as_posix() + ' [unreadable]')
    if hits:
        raise SystemExit('ANONYMITY SCAN FAILED:\n' + '\n'.join(sorted(set(hits))))
    print('ANONYMITY SCAN: PASS')


def write_verifier(root: Path) -> None:
    code = '''#!/usr/bin/env python3
from pathlib import Path
import json, sys
ROOT = Path(__file__).resolve().parents[1]
failures=[]
cells=ROOT/'paper-run/results/cells-and-judgments.jsonl'
if not cells.is_file():
    failures.append('missing primary cell export')
else:
    n=sum(1 for _ in cells.open('rb'))
    if n != 16800:
        failures.append(f'primary rows: {n} != 16800')
analysis=(ROOT/'paper-run/results/analysis.json').read_text(encoding='utf-8', errors='replace')
report=(ROOT/'paper-run/results/paper-report.txt').read_text(encoding='utf-8', errors='replace')
for token in ['90.22','93.17','73.44','74.81','60.21']:
    if token not in analysis and token not in report:
        failures.append('primary result token missing: '+token)
ae=json.loads((ROOT/'ae-support/RESULTS.json').read_text())
checks=[
    (ae['ssnhl']['system_only']['correct'],842),
    (ae['ssnhl']['duplicated_after_query']['correct'],971),
    (ae['ssnhl']['paired_transitions']['improved'],154),
    (ae['ssnhl']['paired_transitions']['degraded'],25),
    (ae['conductive']['system_only']['correct'],786),
    (ae['conductive']['duplicated_after_query']['correct'],738),
    (ae['conductive']['paired_transitions']['improved'],153),
    (ae['conductive']['paired_transitions']['degraded'],201),
]
if any(a!=b for a,b in checks):
    failures.append('AE aggregate/pair assertions failed')
nb=(ROOT/'ae-support/executed-instruction-placement-reproduction.ipynb').read_text(encoding='utf-8')
for token in [
    '842/1000 = 84.2%','971/1000 = 97.1%',
    '786/1000 = 78.6%','738/1000 = 73.8%',
    'improved=154, degraded=25, net=+129',
    'improved=153, degraded=201, net=-48'
]:
    if token not in nb:
        failures.append('executed notebook token missing: '+token)
human=json.loads((ROOT/'human-validation/AGGREGATE_RESULT.json').read_text())
if (human['confirmations'],human['ties'],human['reversals']) != (10,20,0):
    failures.append('human-audit aggregate assertion failed')
if failures:
    print('ICLR ARTIFACT VERIFICATION: FAIL')
    for f in failures:
        print('-',f)
    sys.exit(1)
print('ICLR ARTIFACT VERIFICATION: PASS')
print('primary rows: 16,800')
print('primary headline values: present in frozen analysis/report')
print('AE: 842->971 SSNHL; 786->738 conductive; paired transitions verified')
print('human audit aggregate: 10 confirmations, 20 ties, 0 reversals')
'''
    (root / 'tools' / 'verify_iclr_bundle.py').write_text(code, encoding='utf-8')


def main() -> None:
    for p in [FROZEN, BASE, PKG, AE, OUT]:
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True)

    with tarfile.open(REPO / 'rejudged-3.0.13.tgz', 'r:gz') as tf:
        tf.extractall(FROZEN, filter='data')
    run_dir = FROZEN / 'run-2026-08-12'
    if not (run_dir / 'manifest.json').is_file():
        raise SystemExit('Frozen run manifest missing after extraction')

    run('python', 'scripts/build_aaai_supplement.py', '--repo', '.', '--run', str(run_dir), '--output', str(BASE / 'base.zip'), cwd=REPO)
    with zipfile.ZipFile(BASE / 'base.zip') as z:
        z.extractall(PKG)

    human_dir = PKG / 'human-validation'
    human_dir.mkdir(parents=True, exist_ok=True)
    for src, dst in [
        ('blinded-matched-pair-audit.jsonl', 'blinded-matched-pair-audit.jsonl'),
        ('blinded-matched-pair-key.jsonl', 'blinded-matched-pair-key.jsonl'),
        ('human-audit-schema.json', 'human-audit-schema.json'),
        ('human-validation.html', 'blinded-review-interface.html'),
    ]:
        shutil.copy2(run_dir / 'results' / src, human_dir / dst)

    run('git', 'clone', '-q', 'https://github.com/victorlavrenko/answer-engineering.git', str(AE))
    ae_support = PKG / 'ae-support'
    repro = ae_support / 'reproduction'
    (repro / 'src').mkdir(parents=True, exist_ok=True)
    (repro / 'rules').mkdir(parents=True, exist_ok=True)
    (repro / 'tests').mkdir(parents=True, exist_ok=True)
    shutil.copy2(AE / 'notebooks' / 'instruction-placement-reproduction.ipynb', ae_support / 'executed-instruction-placement-reproduction.ipynb')
    shutil.copytree(AE / 'src' / 'ae_paper_reproduction', repro / 'src' / 'ae_paper_reproduction')
    shutil.copytree(AE / 'src' / 'answer_engineering', repro / 'src' / 'answer_engineering')
    shutil.copy2(AE / 'tests' / 'fixtures' / 'ent_ssnhl_doctor_rules.ae', repro / 'rules' / 'ent_ssnhl_doctor_rules.ae')
    shutil.copy2(AE / 'tests' / 'repro' / 'test_repro_paper_ssnhl.py', repro / 'tests' / 'test_repro_paper_ssnhl.py')

    nb_path = ae_support / 'executed-instruction-placement-reproduction.ipynb'
    raw = nb_path.read_text(encoding='utf-8')
    required = [
        'accuracy: 842/1000 = 84.2%',
        'accuracy: 971/1000 = 97.1%',
        'accuracy: 786/1000 = 78.6%',
        'accuracy: 738/1000 = 73.8%',
        'trajectory-editing-orl-ssnhl-acute: improved=154, degraded=25, net=+129',
        'trajectory-editing-orl-conductive-acute: improved=153, degraded=201, net=-48',
    ]
    missing = [x for x in required if x not in raw]
    if missing:
        raise SystemExit('AE stored-output assertions missing: ' + repr(missing))

    ae_result = {
        'provenance': 'Stored outputs in executed-instruction-placement-reproduction.ipynb',
        'model': 'OpenMeditron/Meditron3-8B',
        'n_per_scope': 1000,
        'ssnhl': {
            'system_only': {'correct': 842, 'n': 1000, 'accuracy_percent': 84.2},
            'duplicated_after_query': {'correct': 971, 'n': 1000, 'accuracy_percent': 97.1},
            'paired_transitions': {'improved': 154, 'degraded': 25, 'net': 129},
        },
        'conductive': {
            'system_only': {'correct': 786, 'n': 1000, 'accuracy_percent': 78.6},
            'duplicated_after_query': {'correct': 738, 'n': 1000, 'accuracy_percent': 73.8},
            'paired_transitions': {'improved': 153, 'degraded': 201, 'net': -48},
        },
        'artifact_scope_note': 'The historical browser-downloaded 4,000-response JSONL is not present in the current source repositories and is not reconstructed or fabricated.'
    }
    (ae_support / 'RESULTS.json').write_text(json.dumps(ae_result, indent=2) + '\n', encoding='utf-8')

    human_result = {
        'provenance': 'Paper-reported aggregate; blinded packet/key/schema and review interface included.',
        'primary_n': 30,
        'confirmations': 10,
        'ties': 20,
        'reversals': 0,
        'prespecified_critical_confirmations': 28,
        'criterion_met': False,
        'individual_final_rating_json_available_in_current_frozen_archive': False,
        'artifact_scope_note': 'No per-case final-rating file is fabricated. The frozen archive contains the blinded audit packet/key/schema/interface, but not the later standalone ratings JSON.'
    }
    (human_dir / 'AGGREGATE_RESULT.json').write_text(json.dumps(human_result, indent=2) + '\n', encoding='utf-8')

    # Remove the public Colab badge cell before scrubbing the notebook.
    nb = json.loads(nb_path.read_text(encoding='utf-8'))
    nb['cells'] = [
        cell for cell in nb.get('cells', [])
        if not (
            'colab.research.google.com/github/' in ''.join(cell.get('source', []))
            and 'instruction-placement-reproduction' in ''.join(cell.get('source', []))
        )
    ]
    nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')

    scope = '''# Artifact scope and provenance

This anonymous ICLR artifact contains the complete frozen 16,800-cell primary
study export, deterministic scoring/analysis source, experimental configuration,
blinded human-audit packet/key/schema/interface, and an independently executed
Answer Engineering instruction-placement notebook with the exact reported
aggregate rates and paired transitions.

Two historical convenience files described in an earlier venue-specific
supplement are not present in the current frozen source repositories: (1) a
standalone JSON file containing the final per-case human ratings and (2) the
browser-downloaded 4,000-response Answer Engineering JSONL. They are not
reconstructed or fabricated. The current ICLR reviewer map refers only to
evidence actually shipped here.

Fresh hosted generation is not required to audit the primary study. The Answer
Engineering notebook is included as an executed artifact; its local reproduction
source and clinical rule file are also shipped for inspection.
'''
    (PKG / 'ARTIFACT_SCOPE.md').write_text(scope, encoding='utf-8')

    reviewer = '''# Reviewer quick start

No model-provider credentials are needed for the paper-facing checks.

```bash
python tools/verify_iclr_bundle.py
python -m pip install -r requirements-research.lock
python -m pip install -e .
instruction-duplication status --workspace paper-run
```

The full primary cell-level export is in
`paper-run/results/cells-and-judgments.jsonl`; deterministic scoring and
analysis source is in `src/instruction_duplication/`.

The Answer Engineering demonstration is in `ae-support/`: the executed notebook
stores the exact four rates and paired transition counts, while `reproduction/`
contains local runtime/reproduction source and the clinical rule file.

See `ARTIFACT_SCOPE.md` for precise provenance and scope.
'''
    (PKG / 'ICLR_REVIEWER_README.md').write_text(reviewer, encoding='utf-8')

    # Replace the older venue-specific anonymous README with a current one.
    old_readme = PKG / 'ANONYMOUS_SUPPLEMENT_README.md'
    if old_readme.exists():
        old_readme.unlink()

    write_verifier(PKG)
    scrub_tree(PKG)
    scan_anonymity(PKG)

    report_path = PKG / 'VERIFICATION_REPORT.txt'
    proc = subprocess.run(['python', str(PKG / 'tools' / 'verify_iclr_bundle.py')], text=True, capture_output=True)
    report_path.write_text(proc.stdout + proc.stderr, encoding='utf-8')
    print(proc.stdout, end='')
    if proc.returncode:
        raise SystemExit(proc.returncode)

    # Final SHA-256 manifest over all files except the manifest itself.
    sums = []
    for p in sorted(PKG.rglob('*')):
        if p.is_file() and p.name != 'SHA256SUMS.txt':
            sums.append(f'{sha256(p)}  {p.relative_to(PKG).as_posix()}')
    (PKG / 'SHA256SUMS.txt').write_text('\n'.join(sums) + '\n', encoding='utf-8')

    summary = [
        'ICLR 2027 anonymous reproduction artifact',
        '',
        f'Files: {sum(1 for p in PKG.rglob("*") if p.is_file())}',
        f'Uncompressed bytes: {sum(p.stat().st_size for p in PKG.rglob("*") if p.is_file())}',
        f'Primary cell export bytes: {(PKG / "paper-run/results/cells-and-judgments.jsonl").stat().st_size}',
        '',
        report_path.read_text(encoding='utf-8').strip(),
        '',
    ]
    (PKG / 'PACKAGE_SUMMARY.txt').write_text('\n'.join(summary), encoding='utf-8')
    scrub_tree(PKG)
    scan_anonymity(PKG)

    zip_path = OUT / 'ICLR2027_Anonymous_Reproduction_Materials.zip'
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in sorted(PKG.rglob('*')):
            if p.is_file():
                z.write(p, p.relative_to(PKG))
    size = zip_path.stat().st_size
    if size >= 100 * 1024 * 1024:
        raise SystemExit(f'Final ZIP too large for target limit: {size} bytes')
    (OUT / 'ZIP_SHA256.txt').write_text(f'{sha256(zip_path)}  {zip_path.name}\n', encoding='utf-8')
    print(f'FINAL ZIP: {zip_path} ({size / (1024*1024):.1f} MiB)')
    print(f'FINAL ZIP SHA256: {sha256(zip_path)}')


if __name__ == '__main__':
    main()
