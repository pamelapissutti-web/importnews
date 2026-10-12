# Trilha original + efeitos sincronizados com o vídeo + mixagem com a narração.
# Uso: python mix.py <pasta_vo> <saida.wav>
# Tudo é sintetizado aqui (sem samples), então a trilha é livre de direitos.
import json, subprocess, sys
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
from scipy import signal
from scipy.ndimage import maximum_filter1d, uniform_filter1d

SR = 48000
DUR = 30.0
N = int(SR * DUR)
rng = np.random.default_rng(7)

VO_DIR = sys.argv[1] if len(sys.argv) > 1 else 'vo_pf_dora'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'mix.wav'

# ── grade musical: 10 compassos de 2,2 s entre o corte do gancho (3,6 s) e o encerramento (25,6 s) ──
BEAT = 0.55                     # 109 BPM
BAR = 4 * BEAT
T0 = 3.6                        # 1º tempo do compasso 1 = corte para o painel
CUTS = [3.6, 9.0, 14.6, 20.2, 25.6]
PROG = ['C', 'G', 'Am', 'F'] * 2 + ['C', 'G']   # 10 compassos → resolve em C no 25,6 s
PAD_V = {'C': [55, 60, 64, 67], 'G': [55, 59, 62, 67], 'Am': [57, 60, 64, 69], 'F': [57, 60, 65, 69]}
ROOT = {'C': 36, 'G': 43, 'Am': 45, 'F': 41}
ARP_V = {'C': [72, 76, 79, 84], 'G': [71, 74, 79, 83], 'Am': [72, 76, 81, 84], 'F': [72, 77, 81, 84]}


def hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def tt(d):
    return np.arange(int(d * SR)) / SR


def bus():
    return np.zeros((N, 2))


def place(b, x, t0, g=1.0, pan=0.0):
    """Soma o sinal x (mono ou estéreo) no barramento b a partir de t0, com pan de potência constante."""
    i0 = int(round(t0 * SR))
    if x.ndim == 1:
        th = (pan + 1) * np.pi / 4
        x = np.stack([x * np.cos(th), x * np.sin(th)], 1) * np.sqrt(2)
    if i0 < 0:
        x, i0 = x[-i0:], 0
    if i0 >= N:
        return
    x = x[: N - i0]
    b[i0 : i0 + len(x)] += x * g


def env_adsr(n, a=0.005, d=0.1, s=0.7, r=0.1):
    e = np.ones(n) * s
    ia, idd, ir = int(a * SR), int(d * SR), int(r * SR)
    ia = min(ia, n)
    e[:ia] = np.linspace(0, 1, ia, endpoint=False)
    j = min(ia + idd, n)
    e[ia:j] = np.linspace(1, s, j - ia, endpoint=False)
    if ir > 0:
        e[-ir:] *= np.linspace(1, 0, min(ir, n))
    return e


def saw(f, d, maxh=4500.0, ph=0.0):
    t = tt(d)
    y = np.zeros_like(t)
    for h in range(1, int(maxh // f) + 1):
        y += np.sin(2 * np.pi * f * h * t + ph * h) / h
    return y * 0.55


def lp(x, fc, order=2):
    return signal.sosfilt(signal.butter(order, fc, 'low', fs=SR, output='sos'), x, axis=0)


def hp(x, fc, order=2):
    return signal.sosfilt(signal.butter(order, fc, 'high', fs=SR, output='sos'), x, axis=0)


def bp(x, f1, f2, order=2):
    return signal.sosfilt(signal.butter(order, [f1, f2], 'band', fs=SR, output='sos'), x, axis=0)


def sweep_noise(d, fpath, q=1.2):
    """Ruído com passa-faixa variando no tempo (fpath: função 0..1 → Hz), processado em blocos."""
    n = int(d * SR)
    x = rng.standard_normal(n)
    y = np.zeros(n)
    blk = 256
    zi = np.zeros((1, 2))
    for i in range(0, n, blk):
        fc = float(np.clip(fpath(i / n), 60, SR / 2 * 0.9))
        bw = fc / q
        sos = signal.butter(1, [max(30, fc - bw / 2), min(SR / 2 * 0.95, fc + bw / 2)], 'band', fs=SR, output='sos')
        y[i : i + blk], zi = signal.sosfilt(sos, x[i : i + blk], zi=zi)
    return y


def reverb_ir(rt=1.6, d=2.4):
    t = tt(d)
    dec = np.exp(-6.9 * t / rt)
    ir = np.stack([rng.standard_normal(len(t)) * dec, rng.standard_normal(len(t)) * dec], 1)
    ir = lp(ir, 6500)
    ir[: int(0.012 * SR)] = 0                          # pré-delay
    return ir / np.sqrt((ir ** 2).sum(0)).max() * 0.9


def reverb(b, rt=1.6):
    ir = reverb_ir(rt)
    return np.stack([signal.fftconvolve(b[:, c], ir[:, c])[:N] for c in range(2)], 1)


# ── tempos de kick (para batida e sidechain) ──
def beat_t(bar, beat):
    return T0 + bar * BAR + beat * BEAT


kick_times = []
for bar in range(10):
    for bt in range(4):
        t = beat_t(bar, bt)
        if bar == 9 and bt == 3:
            continue                                   # respiro antes do acorde final
        kick_times.append(t)
for t in [25.6, 25.6 + 2 * BEAT, 25.6 + BAR, 25.6 + BAR + 2 * BEAT]:   # encerramento em meio-tempo
    if t < 29.4:
        kick_times.append(t)

# curva de "pump" (sidechain) aplicada a pad e baixo
tt_all = np.arange(N) / SR
pump = np.ones(N)
for k in kick_times:
    i0 = int(k * SR)
    seg = tt_all[i0 : i0 + int(0.4 * SR)] - k
    pump[i0 : i0 + len(seg)] = np.minimum(pump[i0 : i0 + len(seg)], 1 - 0.55 * np.exp(-seg / 0.11))
pump = uniform_filter1d(pump, int(0.004 * SR))

pad, bass, drums, arp, sfx, ui, send = bus(), bus(), bus(), bus(), bus(), bus(), bus()

# ── PAD ──
def pad_chord(notes, d, bright=2200, det=0.11):
    y = np.zeros(int(d * SR))
    for i, m in enumerate(notes):
        for k, dc in enumerate((-det, 0, det)):
            y += saw(hz(m) * 2 ** (dc / 12), d, ph=rng.uniform(0, 6.28))
    y = lp(y, bright, 2) / (len(notes) * 3) * 2.2
    return y

# intro (0–3,6 s): Lá menor escuro abrindo o filtro
intro = pad_chord([57, 60, 64, 69], 3.9, bright=900)
intro_b = pad_chord([57, 60, 64, 69], 3.9, bright=2600)
xf = np.clip(tt(3.9) / 3.4, 0, 1) ** 1.5
intro = (intro * (1 - xf) + intro_b * xf) * env_adsr(len(intro), a=0.6, d=0.1, s=1, r=0.35)
place(pad, intro, 0.0, 0.85, -0.15)
place(pad, np.roll(intro, 900), 0.0, 0.85, 0.15)
for bar, ch in enumerate(PROG):
    t = beat_t(bar, 0)
    y = pad_chord(PAD_V[ch], BAR + 0.35) * env_adsr(int((BAR + 0.35) * SR), a=0.06, d=0.2, s=0.85, r=0.32)
    place(pad, y, t - 0.02, 0.8, -0.2)
    place(pad, np.roll(y, 700), t - 0.02, 0.8, 0.2)
# acorde final (Cadd9) com cauda longa
fin = pad_chord([48, 55, 62, 64, 67, 72], 4.4, bright=3200) * env_adsr(int(4.4 * SR), a=0.01, d=0.6, s=0.7, r=1.6)
place(pad, fin, 25.6, 1.1, -0.15)
place(pad, np.roll(fin, 800), 25.6, 1.1, 0.15)
pad *= pump[:, None] * 0.5 + 0.5

# ── BAIXO: colcheias no contratempo + nota longa grave no 1º tempo ──
def bass_note(m, d, g=1.0):
    t = tt(d)
    f = hz(m)
    y = np.sin(2 * np.pi * f * t) + 0.35 * np.sin(4 * np.pi * f * t) + 0.12 * np.sin(6 * np.pi * f * t)
    y = np.tanh(1.6 * y) * env_adsr(len(t), a=0.004, d=0.12, s=0.55, r=0.05)
    return lp(y, 900) * g

for bar, ch in enumerate(PROG):
    r = ROOT[ch]
    for e in range(8):
        t = beat_t(bar, e / 2)
        if bar == 9 and e >= 6:
            continue
        if e % 2 == 1:
            place(bass, bass_note(r, 0.24), t, 0.55)
        elif bar >= 2:                                 # a partir da cena 3 o baixo também marca o tempo
            place(bass, bass_note(r - 12 if r >= 41 else r, 0.2), t, 0.35)
place(bass, bass_note(36, 2.6) * env_adsr(int(2.6 * SR), a=0.005, d=0.4, s=0.6, r=1.0), 25.6, 0.7)
bass *= pump[:, None] * 0.6 + 0.4

# ── BATERIA ──
def kick():
    t = tt(0.42)
    f = 44 + 90 * np.exp(-t / 0.04)
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.15)
    click = hp(rng.standard_normal(len(t)), 2500) * np.exp(-t / 0.003) * 0.25
    return np.tanh(1.4 * (y + click))


def hat(d=0.045):
    t = tt(d + 0.05)
    return hp(rng.standard_normal(len(t)), 7500, 4) * np.exp(-t / d)


def clap():
    t = tt(0.35)
    n = bp(rng.standard_normal(len(t)), 900, 3200)
    e = np.zeros(len(t))
    for o in (0, 0.009, 0.018):
        i = int(o * SR)
        e[i:] += np.exp(-(t[: len(t) - i]) / 0.006)
    e += 0.6 * np.exp(-t / 0.11) * (t > 0.018)
    return n * e * 0.6


def snare_roll_hit(g):
    t = tt(0.16)
    return (bp(rng.standard_normal(len(t)), 1200, 6000) * np.exp(-t / 0.05) + 0.3 * np.sin(2 * np.pi * 190 * t) * np.exp(-t / 0.04)) * g


def crash(d=2.2):
    t = tt(d)
    return hp(rng.standard_normal(len(t)), 4500, 2) * np.exp(-t / 0.7) * 0.5


KICK = kick()
for k in kick_times:
    place(drums, KICK, k, 0.95)
for bar in range(10):
    for e in range(8):
        t = beat_t(bar, e / 2)
        if bar == 9 and e >= 5:
            continue
        if e % 2 == 1:
            open_h = (e == 7 and bar % 2 == 1)
            place(drums, hat(0.16 if open_h else 0.045), t, 0.22 if open_h else 0.28, 0.25)
    if bar >= 5:                                       # cena 4 em diante: palmas no 2 e 4 + chocalho em semicolcheias
        for bt in (1, 3):
            if bar == 9 and bt == 3:
                continue
            c = clap()
            place(drums, c, beat_t(bar, bt), 0.55)
            place(send, c, beat_t(bar, bt), 0.35)
        for s16 in range(16):
            if bar == 9 and s16 >= 8:
                continue
            place(drums, hat(0.02), beat_t(bar, s16 / 4), 0.07 + 0.05 * (s16 % 2), -0.35)
# virada de caixa crescendo até o acorde final (24,5 → 25,6 s)
t_roll = beat_t(9, 2)
for i in range(8):
    tr = t_roll + i * BEAT / 4
    if tr < 25.55:
        place(drums, snare_roll_hit(0.18 + 0.07 * i), tr, 1.0, 0.1 * (i % 2 * 2 - 1))
# encerramento: chimbal leve até 29 s
for e in range(14):
    t = 25.6 + e * BEAT / 2
    if e % 2 == 1 and t < 29.0:
        place(drums, hat(0.05), t, 0.2, 0.25)
for t in (3.6, 25.6):
    place(drums, crash(), t, 0.7, 0.2)
    place(send, crash(), t, 0.25, -0.2)

# ── ARPEJO (entra na cena 3) com delay pingue-pongue ──
def pluck(m, d=0.5):
    t = tt(d)
    f = hz(m)
    y = (np.sin(2 * np.pi * f * t) * np.exp(-t / 0.22) + 0.45 * np.sin(4 * np.pi * f * t) * np.exp(-t / 0.09)
         + 0.2 * np.sin(6 * np.pi * f * t) * np.exp(-t / 0.05))
    y[: int(0.002 * SR)] *= np.linspace(0, 1, int(0.002 * SR))
    return y

PAT = [0, 2, 1, 3, 2, 1, 3, 2, 0, 2, 1, 3, 2, 3, 1, 2]
for bar, ch in enumerate(PROG):
    if bar < 2:
        continue
    for s16 in range(16):
        if bar == 9 and s16 >= 12:
            continue
        m = ARP_V[ch][PAT[s16]]
        g = 0.16 if s16 % 4 == 0 else 0.11
        if bar >= 8:
            g *= 1.15
        place(arp, pluck(m), beat_t(bar, s16 / 4), g, -0.3 if s16 % 2 else 0.3)
dly = int(0.75 * BEAT * SR)                            # colcheia pontuada
echo = np.zeros_like(arp)
src = arp.copy()
for k in range(1, 5):
    sh = np.zeros_like(arp)
    sh[k * dly :] = src[: N - k * dly]
    sh = sh[:, ::-1] if k % 2 else sh                  # alterna lados
    echo += sh * (0.38 ** k)
arp = arp + lp(echo, 3500)
place(send, arp, 0, 0.3)

# ── EFEITOS: riser no gancho, whoosh + grave em cada corte, sons de interface ──
riser_d = 1.75
r = sweep_noise(riser_d, lambda p: 300 + 7000 * p ** 2, q=1.6) * np.linspace(0, 1, int(riser_d * SR)) ** 2
tone = np.sin(2 * np.pi * np.cumsum(220 + 660 * tt(riser_d) / riser_d * tt(riser_d) / riser_d) / SR) * np.linspace(0, 1, int(riser_d * SR)) ** 3 * 0.25
place(sfx, r * 0.5 + tone, 3.6 - riser_d, 0.6)


def whoosh(d=0.8):
    y = sweep_noise(d, lambda p: 350 + 4200 * np.sin(np.pi * min(1, p * 1.05)) ** 1.5, q=1.4)
    e = np.interp(tt(d), [0, d * 0.48, d * 0.55, d], [0, 1, 0.9, 0])
    return y * e ** 1.4


def boom():
    t = tt(0.9)
    f = 42 + 50 * np.exp(-t / 0.07)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.28)


for c in CUTS:
    w = whoosh()
    wl = w * np.linspace(1.2, 0.5, len(w))
    wr = w * np.linspace(0.5, 1.2, len(w))
    place(sfx, np.stack([wl, wr], 1), c - 0.4, 0.42 if c not in (3.6, 25.6) else 0.3)
    place(sfx, boom(), c, 0.55 if c in (3.6, 25.6) else 0.32)
    place(send, np.stack([wl, wr], 1), c - 0.4, 0.15)


def blip(m, g=1.0, d=0.35):
    t = tt(d)
    f = hz(m)
    y = (np.sin(2 * np.pi * f * t) + 0.25 * np.sin(2 * np.pi * 2 * f * t) * np.exp(-t / 0.03)) * np.exp(-t / 0.09)
    y[: int(0.002 * SR)] *= np.linspace(0, 1, int(0.002 * SR))
    return y * g

# cena 1: lançamentos surgindo (pentatônica de Lá menor) e brilho do logo
for i, m in enumerate([69, 72, 74, 76, 79, 81, 84, 86]):
    place(ui, blip(m, 0.5), 0.2 + i * 0.11 + 0.06, 1, -0.5 + i * 0.14)
for i, m in enumerate([81, 84, 88, 93]):
    place(ui, blip(m, 0.35, 0.6), 2.5 + i * 0.07, 1, 0.3 - 0.2 * i)
place(ui, blip(74, 0.55) + blip(79, 0.4), 6.95, 1, 0.2)                     # "paga" ✓
for i, m in enumerate([76, 79, 81, 84]):
    place(ui, blip(m, 0.42), 12.25 + i * 0.22, 1, 0.35)                     # categorias sugeridas
for i, m in enumerate([74, 79, 83]):
    place(ui, blip(m, 0.38), 15.6 + i * 0.42, 1, -0.3)                      # rosca
place(ui, blip(76, 0.45) + blip(81, 0.35), 17.78, 1, 0.3)                    # balão de outubro
for i, m in enumerate([72, 76, 79, 84]):
    place(ui, blip(m, 0.36), 21.62 + i * 0.2, 1, 0.3)                       # caixinhas
place(ui, blip(86, 0.38), 23.92, 1, 0)                                       # Open Finance
for i, m in enumerate([67, 72, 76, 79]):
    place(ui, blip(m, 0.32), 26.55 + i * 0.15 if i < 3 else 27.17, 1, -0.2 + 0.2 * i)   # benefícios + botão
for i, m in enumerate([88, 91, 96]):
    place(ui, blip(m, 0.22, 0.5), 27.95 + i * 0.06, 1, 0.4)                 # brilho no botão
place(send, ui, 0, 0.45)
place(send, pad, 0, 0.18)

# ── soma da trilha ──
music = pad * 0.55 + bass * 0.75 + drums * 0.62 + arp * 0.5 + sfx * 0.55 + ui * 0.35
music += reverb(send, 1.7) * 0.28
music = hp(music, 28)
fade = np.ones(N)
fi = int(29.1 * SR)
fade[fi:] = np.linspace(1, 0, N - fi) ** 1.6
music *= fade[:, None]

# ── narração: monta a faixa, trata no ffmpeg (EQ, compressão, de-esser) ──
meta = json.load(open(f'{VO_DIR}/meta.json'))
vo = np.zeros(N)
for m in meta:
    a, sr = sf.read(f'{VO_DIR}/l{m["i"]}.wav', dtype='float64')
    a = signal.resample_poly(a, SR, sr)
    i0 = int(m['t0'] * SR)
    a = a[: N - i0]
    vo[i0 : i0 + len(a)] += a
sf.write('vo_raw.wav', vo, SR, subtype='FLOAT')
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', 'vo_raw.wav', '-af',
                'highpass=f=85,equalizer=f=220:t=q:w=1.1:g=-2,equalizer=f=3200:t=q:w=1.3:g=2.5,equalizer=f=9500:t=q:w=1:g=1.5,'
                'acompressor=threshold=-24dB:ratio=3:attack=4:release=90:makeup=3dB,deesser=i=0.4',
                '-ar', str(SR), '-c:a', 'pcm_f32le', 'vo_proc.wav'], check=True)
vo, _ = sf.read('vo_proc.wav', dtype='float64')
vo = vo[:N] if len(vo) >= N else np.pad(vo, (0, N - len(vo)))

# ── loudness: voz em -16 LUFS; trilha ~11 LU abaixo; ducking de ~6 dB enquanto há fala ──
meter = pyln.Meter(SR)
vo *= 10 ** ((-16 - meter.integrated_loudness(np.stack([vo, vo], 1))) / 20)   # medida como sai na mix (nos 2 canais)
music *= 10 ** ((-22 - meter.integrated_loudness(music)) / 20)
lvl = np.sqrt(np.maximum(uniform_filter1d(vo ** 2, int(0.05 * SR)), 0))
talk = (20 * np.log10(lvl + 1e-9) > -42).astype(float)
talk = maximum_filter1d(talk, int(0.35 * SR))                     # segura entre palavras
talk = uniform_filter1d(talk, int(0.18 * SR))                     # rampas suaves
duck = 10 ** (-6.5 * talk / 20)
music_d = music * duck[:, None]

mix = music_d + vo[:, None]
sf.write('music_stem.wav', music_d.astype(np.float32), SR, subtype='FLOAT')
sf.write(OUT, mix.astype(np.float32), SR, subtype='FLOAT')
print('voz LUFS', round(meter.integrated_loudness(np.stack([vo, vo], 1)), 1), '· trilha (com ducking)', round(meter.integrated_loudness(music_d), 1),
      '· mix', round(meter.integrated_loudness(mix), 1), '· pico', round(20 * np.log10(np.abs(mix).max()), 1), 'dBFS')
