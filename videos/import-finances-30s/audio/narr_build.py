# Uso: ./venv/bin/python narr_build.py <gravacao.ogg|m4a|wav> vo_narracao && ./venv/bin/python mix.py vo_narracao mix.wav && ./master.sh mix.wav ../import-finances-30s-sem-audio.mp4 final.mp4
# Encaixa a narração gravada (uma tomada com as 6 falas) no tempo das cenas.
# 1) limpeza da tomada inteira (de-clip, passa-altas, redução de ruído)
# 2) recorte de cada fala, com pausas internas encurtadas
# 3) se ainda passar da janela da cena, acelera com rubberband (tom preservado)
# Saída: <saida>/l1..l6.wav + meta.json no formato que o mix.py lê.
import json, os, subprocess, sys
import numpy as np, soundfile as sf
from scipy import signal
from scipy.ndimage import uniform_filter1d

SRC = sys.argv[1]
OUT = sys.argv[2]
os.makedirs(OUT, exist_ok=True)
SR = 48000

# trechos da tomada que contêm cada fala (achados pelas pausas + transcrição)
CHUNKS = [(1.32, 5.88), (5.88, 12.05), (12.05, 16.78), (16.78, 21.79), (21.79, 26.55), (26.55, 30.78)]
# início de cada fala no vídeo e fim máximo (deixa respiro antes da próxima)
SLOTS = [(0.00, 3.92), (4.06, 9.32), (9.48, 14.55), (14.75, 20.15), (20.35, 25.55), (25.75, 29.75)]
MAX_PAUSE = [0.12, 0.16, 0.16, 0.16, 0.16, 0.20]
PRE, POST = 0.06, 0.14

subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', SRC, '-ac', '1', '-ar', str(SR),
                '-af', 'adeclip,highpass=f=80:poles=2,afftdn=nr=10:nf=-50:tn=1',
                '-c:a', 'pcm_f32le', f'{OUT}/clean.wav'], check=True)
a, _ = sf.read(f'{OUT}/clean.wav', dtype='float64')
hop = int(0.01 * SR)
db = 20 * np.log10(np.sqrt(np.maximum(uniform_filter1d(a ** 2, int(0.025 * SR)), 0))[::hop] + 1e-9)


def xfade_join(parts, ms=15):
    n = int(ms / 1000 * SR)
    out = parts[0]
    for p in parts[1:]:
        r = np.linspace(0, 1, n)
        out = np.concatenate([out[:-n], out[-n:] * (1 - r) + p[:n] * r, p[n:]])
    return out


meta = []
for k, ((c0, c1), (t0, t1)) in enumerate(zip(CHUNKS, SLOTS)):
    seg = db[int(c0 * 100): int(c1 * 100)]
    sp = seg > np.percentile(seg, 95) - 30
    idx = np.where(sp)[0]
    on, off = c0 + idx[0] * 0.01 - PRE, c0 + idx[-1] * 0.01 + POST
    # pausas internas (≥ 80 ms abaixo do limiar) → limitadas a MAX_PAUSE, cortando o miolo
    cuts, i = [], idx[0]
    while i <= idx[-1]:
        if not sp[i]:
            j = i
            while not sp[j]:
                j += 1
            g0, g1 = c0 + i * 0.01, c0 + j * 0.01
            if g1 - g0 > MAX_PAUSE[k]:
                keep = MAX_PAUSE[k] / 2
                cuts.append((g0 + keep, g1 - keep))
            i = j
        else:
            i += 1
    parts, cur = [], on
    for x0, x1 in cuts:
        parts.append(a[int(cur * SR): int(x0 * SR)])
        cur = x1
    parts.append(a[int(cur * SR): int(off * SR)])
    y = xfade_join(parts)
    win = t1 - t0
    dur = len(y) / SR - POST          # a cauda pode invadir o respiro antes da próxima fala
    tempo = 1.0
    if dur > win:
        tempo = dur / win
        sf.write(f'{OUT}/tmp.wav', y, SR, subtype='FLOAT')
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', f'{OUT}/tmp.wav', '-af',
                        f'rubberband=tempo={tempo:.4f}:pitchq=quality:formant=preserved:window=short:transients=mixed',
                        '-c:a', 'pcm_f32le', f'{OUT}/tmp2.wav'], check=True)
        y, _ = sf.read(f'{OUT}/tmp2.wav', dtype='float64')
    # rampas curtas nas pontas para não estalar
    fi, fo = int(0.012 * SR), int(0.08 * SR)
    y[:fi] *= np.linspace(0, 1, fi)
    y[-fo:] *= np.linspace(1, 0, fo)
    sf.write(f'{OUT}/l{k + 1}.wav', y, SR, subtype='FLOAT')
    meta.append({'i': k + 1, 't0': t0, 'win': round(win, 2), 'dur': round(len(y) / SR, 2), 'tempo': round(tempo, 3),
                 'orig': round(off - on, 2), 'pauses_cut': round(sum(x1 - x0 for x0, x1 in cuts), 2)})
    print(f'L{k + 1}: original {off - on:.2f}s · pausas −{sum(x1 - x0 for x0, x1 in cuts):.2f}s · '
          f'tempo ×{tempo:.3f} → {len(y) / SR:.2f}s (janela {win:.2f}s, entra em {t0:.2f}s)')
for f in ('tmp.wav', 'tmp2.wav'):
    if os.path.exists(f'{OUT}/{f}'):
        os.remove(f'{OUT}/{f}')
json.dump(meta, open(f'{OUT}/meta.json', 'w'), ensure_ascii=False, indent=1)
