#!/bin/bash
# master.sh <mix.wav> <video_mudo.mp4> <saida.mp4> — loudnorm em 2 passagens (-14 LUFS, pico -1,5 dBTP) e mux com o vídeo
set -euo pipefail
IN=$1; VID=$2; OUT=$3
J=$(ffmpeg -hide_banner -i "$IN" -af loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json -f null - 2>&1 | sed -n '/^{/,/^}/p')
g() { echo "$J" | python3 -c "import json,sys;print(json.load(sys.stdin)['$1'])"; }
ffmpeg -y -loglevel error -i "$IN" -af "loudnorm=I=-14:TP=-1.5:LRA=11:measured_I=$(g input_i):measured_TP=$(g input_tp):measured_LRA=$(g input_lra):measured_thresh=$(g input_thresh):offset=$(g target_offset):linear=true,aresample=48000" -c:a pcm_s16le "${OUT%.mp4}.master.wav"
ffmpeg -y -loglevel error -i "$VID" -i "${OUT%.mp4}.master.wav" -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -b:a 192k -ar 48000 -shortest -movflags +faststart "$OUT"
ffmpeg -hide_banner -i "$OUT" -af ebur128=peak=true -f null - 2>&1 | grep -E "^\s+(I:|Peak:|LRA:)" | tr -s ' '
