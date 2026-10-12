// Renderiza video.html quadro a quadro (seek determinístico) e codifica em MP4 com ffmpeg.
// Uso: node render.cjs frames <t1,t2,...>   → PNGs de conferência em ./shots
//      node render.cjs video [saida.mp4]    → vídeo completo 1080x1920 @ 30 fps
const { chromium } = require('playwright');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const FPS = 30;
const HTML = 'file://' + path.join(__dirname, 'video.html');

async function open() {
  const browser = await chromium.launch({ args: ['--font-render-hinting=none', '--disable-lcd-text'] });
  const page = await browser.newPage({ viewport: { width: 1080, height: 1920 }, deviceScaleFactor: 1 });
  await page.goto(HTML);
  await page.waitForSelector('body[data-ready="1"]');
  return { browser, page };
}

async function frames(times) {
  const { browser, page } = await open();
  fs.mkdirSync(path.join(__dirname, 'shots'), { recursive: true });
  for (const t of times) {
    await page.evaluate((x) => window.seek(x), t);
    await page.screenshot({ path: path.join(__dirname, 'shots', `t${String(t).padStart(5, '0')}.png`) });
  }
  await browser.close();
}

async function video(out) {
  const { browser, page } = await open();
  const dur = await page.evaluate(() => window.DUR);
  const n = Math.round(dur * FPS);
  const ff = spawn('ffmpeg', ['-y', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', String(FPS), '-c:v', 'png', '-i', '-',
    '-c:v', 'libx264', '-preset', 'slow', '-crf', '16', '-pix_fmt', 'yuv420p', '-profile:v', 'high', '-level', '4.2',
    '-r', String(FPS), '-movflags', '+faststart', out], { stdio: ['pipe', 'inherit', 'inherit'] });
  const t0 = Date.now();
  for (let i = 0; i < n; i++) {
    await page.evaluate((x) => window.seek(x), i / FPS);
    const buf = await page.screenshot({ type: 'png' });
    if (!ff.stdin.write(buf)) await new Promise((r) => ff.stdin.once('drain', r));
    if (i % 90 === 0) console.log(`quadro ${i}/${n} · ${((Date.now() - t0) / 1000).toFixed(0)}s`);
  }
  ff.stdin.end();
  await new Promise((r) => ff.on('close', r));
  await browser.close();
  console.log('ok', out, `${((Date.now() - t0) / 1000).toFixed(0)}s`);
}

const [mode, arg] = process.argv.slice(2);
if (mode === 'frames') frames(arg.split(',').map(Number));
else video(arg || path.join(__dirname, 'import-finances-30s.mp4'));
