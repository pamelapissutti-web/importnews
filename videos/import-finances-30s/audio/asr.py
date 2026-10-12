# Transcreve as falas geradas (Whisper small, pt) para conferir a pronúncia sem precisar ouvir.
# Modelo: https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-small.tar.bz2 (extrair em asr/)
import sys, json, glob
import numpy as np, soundfile as sf, sherpa_onnx
d = 'asr/sherpa-onnx-whisper-small/'
rec = sherpa_onnx.OfflineRecognizer.from_whisper(encoder=d + 'small-encoder.int8.onnx', decoder=d + 'small-decoder.int8.onnx', tokens=d + 'small-tokens.txt', language='pt', task='transcribe', num_threads=4)
for f in sys.argv[1:]:
    a, sr = sf.read(f, dtype='float32')
    if a.ndim > 1: a = a.mean(1)
    s = rec.create_stream(); s.accept_waveform(sr, a); rec.decode_stream(s)
    print(f'{f}: {s.result.text.strip()}')
