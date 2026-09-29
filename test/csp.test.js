'use strict';
// CSP connect-src regression test: the Model Cell fetches the Hugging Face
// Hub API (https://huggingface.co/api/models) directly from the browser, so
// connect-src must allow it. Guards against regressions that re-block the
// fetch with only `connect-src 'self' ...jsdelivr`.
//
// Asserts against the live Helmet response header (not the source text):
//   - connect-src includes 'self', https://cdn.jsdelivr.net (Pyodide),
//     and https://huggingface.co (Model Cell Hub API)
//   - connect-src contains no `*`
//   - default-src 'self' and the existing script-src origins are intact
const assert = require('assert');
const http = require('http');

const app = require('../server.js');

function getHeader(port, path) {
  return new Promise((resolve, reject) => {
    http.get({ host: '127.0.0.1', port, path }, (res) => {
      res.resume();
      res.on('end', () => resolve(res.headers));
    }).on('error', reject);
  });
}

function directiveValue(csp, name) {
  const m = String(csp || '').split(';').map((s) => s.trim())
    .find((s) => s === name || s.startsWith(name + ' '));
  assert(m, `CSP header must contain a ${name} directive: ${csp}`);
  return m.slice(name.length).trim();
}

async function main() {
  const server = await new Promise((resolve) => {
    const s = app.listen(0, '127.0.0.1', () => resolve(s));
  });
  try {
    const port = server.address().port;
    const headers = await getHeader(port, '/');
    const csp = headers['content-security-policy'];
    assert(csp, 'server must emit a Content-Security-Policy header');

    const connect = directiveValue(csp, 'connect-src').split(/\s+/);
    for (const origin of ["'self'", 'https://cdn.jsdelivr.net', 'https://huggingface.co']) {
      assert(connect.includes(origin), `connect-src must include ${origin} (got: ${connect.join(' ')})`);
    }
    assert(!connect.includes('*'), `connect-src must not use * (got: ${connect.join(' ')})`);
    console.log(`✓ connect-src allows Hub API: ${connect.join(' ')}`);

    const def = directiveValue(csp, 'default-src').split(/\s+/);
    assert(def.includes("'self'"), 'default-src must keep \'self\'');
    console.log('✓ default-src keeps \'self\'');

    const script = directiveValue(csp, 'script-src').split(/\s+/);
    for (const origin of ["'self'", "'wasm-unsafe-eval'", 'https://cdn.jsdelivr.net', 'https://cdnjs.cloudflare.com']) {
      assert(script.includes(origin), `script-src must keep ${origin} (got: ${script.join(' ')})`);
    }
    assert(!script.includes("'unsafe-eval'"), 'script-src must not broaden to unsafe-eval');
    console.log('✓ script-src intact (wasm-unsafe-eval, no unsafe-eval)');
  } finally {
    server.close();
  }
  console.log('\nAll CSP connect-src tests passed.');
}

main().catch((e) => { console.error('TEST FAILED:', e); process.exit(1); });
