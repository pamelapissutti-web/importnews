# Gera as falas da narração (pt-BR) com Kokoro, uma por cena, ajustando a velocidade para caber na janela.
# Preparação (uma vez):
#   python3 -m venv venv && ./venv/bin/pip install kokoro-onnx soundfile scipy pyloudnorm sherpa-onnx
#   mkdir models && cd models
#   curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
#   curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
# Uso: ./venv/bin/python tts.py pf_dora vo_pf_dora   (vozes pt-BR: pf_dora, pm_alex, pm_santa)
# Depois: ./venv/bin/python mix.py vo_pf_dora mix.wav && ./master.sh mix.wav ../import-finances-30s-sem-audio.mp4 final.mp4
import json, sys, re
import numpy as np, soundfile as sf
from kokoro_onnx import Kokoro

k = Kokoro('models/kokoro-v1.0.onnx', 'models/voices-v1.0.bin')
voice = sys.argv[1] if len(sys.argv) > 1 else 'pf_dora'
outdir = sys.argv[2] if len(sys.argv) > 2 else 'vo'
import os; os.makedirs(outdir, exist_ok=True)

BRAND_IN = 'Ímpórt Fainénssis'
BRAND_PH = 'ˈimpɔɾt faɪnˈɛnsis'
# (início, janela máx., texto)
LINES = [
    (0.15, 3.50, f'Pra onde vai o seu dinheiro? Conheça o {BRAND_IN}.'),
    (3.95, 4.95, 'No painel, você vê o seu mês inteiro: o que entrou, o que saiu e o que falta pagar.'),
    (9.20, 5.20, 'Importe o extrato do seu banco, e cada lançamento já chega categorizado.'),
    (14.80, 5.10, 'Nos gráficos, você vê para onde vai cada real, mês a mês.'),
    (20.40, 5.00, 'E o seu saldo rende sozinho, com base no cê dê i, calculado dia a dia.'),
    (25.85, 3.90, f'{BRAND_IN}. Seu dinheiro sob controle. Comece agora!'),
]
brand_raw = k.tokenizer.phonemize(BRAND_IN, 'pt-br').strip()
meta = []
for i, (t0, win, text) in enumerate(LINES):
    ph = k.tokenizer.phonemize(text, 'pt-br').replace(brand_raw, BRAND_PH)
    speed = 1.0
    for _ in range(8):
        audio, sr = k.create(ph, voice=voice, speed=speed, lang='pt-br', is_phonemes=True)
        dur = len(audio) / sr
        if dur <= win or speed >= 1.3: break
        speed = min(1.3, speed * dur / win * 1.02)
    sf.write(f'{outdir}/l{i+1}.wav', audio, sr)
    meta.append({'i': i + 1, 't0': t0, 'win': win, 'dur': round(dur, 2), 'speed': round(speed, 3), 'text': text, 'ph': ph})
    print(f'{voice} l{i+1}: {dur:.2f}s / janela {win}s · vel {speed:.2f} · {text}')
json.dump(meta, open(f'{outdir}/meta.json', 'w'), ensure_ascii=False, indent=1)
