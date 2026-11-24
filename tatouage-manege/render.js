// Usage: node render.js in.svg out.png [scale]
const { chromium } = require('playwright');
const fs = require('fs');
(async () => {
  const [svgPath, pngPath, scaleArg] = process.argv.slice(2);
  const svg = fs.readFileSync(svgPath, 'utf8');
  const m = svg.match(/viewBox="0 0 ([\d.]+) ([\d.]+)"/);
  const w = Math.ceil(+m[1]), h = Math.ceil(+m[2]);
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: w, height: h }, deviceScaleFactor: +(scaleArg || 2) });
  await page.setContent(`<html><body style="margin:0;background:#fff">${svg}</body></html>`);
  await page.locator('svg').screenshot({ path: pngPath });
  await browser.close();
})();
