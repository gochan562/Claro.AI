/* PyodideLocal — ordinary-Python execution in the browser.
 *
 * Ordinary notebook code cells must run locally (Pyodide/WASM) and never be
 * sent to Modal, ZeroGPU, or any remote GPU provider.  This module provides a
 * lazily-initialized, process-wide Pyodide runtime whose globals persist
 * between cells, with:
 *
 *   - serialized execution (a promise queue) so concurrent Run actions can
 *     never corrupt the shared runtime;
 *   - stdout/stderr capture per run;
 *   - Python tracebacks surfaced as normal cell errors;
 *   - a clear hint when an import is unavailable in the browser runtime.
 *
 * The AI/model-cell path (ZeroGPU/Modal) is completely separate and untouched.
 *
 * UMD: exposes window.PyodideLocal in the browser and module.exports in Node
 * (so it can be unit-tested with a mocked loader).
 */
(function (root, factory) {
  const api = factory();
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  } else {
    root.PyodideLocal = api.defaultRunner;
    root.createPyodideRunner = api.createRunner;
  }
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const PYODIDE_VERSION = '0.26.4';
  const PYODIDE_BASE_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;

  const MISS =
    '\n\n(Pyodide note: some packages are not available in the in-browser ' +
    'Python runtime. Only packages with pure-Python/compiled Pyodide builds ' +
    'can be imported here; see https://pyodide.org/en/stable/usage/packages-in-pyodide.html)';

  function createRunner(options) {
    const opts = options || {};
    const injectScript = opts.injectScript || defaultInjectScript;
    const resolveLoader = opts.resolveLoader || (() => globalThis.loadPyodide);
    // `globals` passthrough exists only so tests can use a custom base URL.
    const loadOptions = opts.loadOptions || { indexURL: PYODIDE_BASE_URL };

    let runtimePromise = null;  // singleton runtime (lazily created)
    let queue = Promise.resolve(); // serializes executions

    async function ensureRuntime() {
      if (runtimePromise) return runtimePromise;
      runtimePromise = (async () => {
        await injectScript(`${PYODIDE_BASE_URL}pyodide.js`);
        const loadPyodide = resolveLoader();
        if (typeof loadPyodide !== 'function') {
          throw new Error('Pyodide loader did not become available.');
        }
        // Await fully: no runPythonAsync before the runtime resolves.
        return await loadPyodide(loadOptions);
      })().catch((err) => {
        // Allow a later retry after a failed download/init.
        runtimePromise = null;
        throw err;
      });
      return runtimePromise;
    }

    async function runOnce(code) {
      const py = await ensureRuntime();
      let stdout = '';
      let stderr = '';
      // Pyodide persists its global namespace across runPythonAsync calls on
      // the same runtime, which is exactly what consecutive cells rely on.
      py.setStdout({ batched: (line) => { stdout += line + '\n'; } });
      py.setStderr({ batched: (line) => { stderr += line + '\n'; } });
      try {
        await py.runPythonAsync(code);
        return { ok: true, stdout, stderr, error: null };
      } catch (err) {
        // A Python exception → Pyodide raises a JS error whose message ends
        // with the Python traceback.  Surface it as a normal cell error.
        let message = (err && err.message) ? String(err.message) : String(err);
        if (/ModuleNotFoundError|ImportError/.test(message)) {
          message += MISS;
        }
        return { ok: false, stdout, stderr, error: message };
      }
    }

    function run(code) {
      // Chain onto the shared queue so two cells can never execute inside the
      // same runtime simultaneously.
      const job = queue.then(() => runOnce(code));
      queue = job.catch(() => {});
      return job;
    }

    return {
      run,
      ensureRuntime,
      isLoaded: () => runtimePromise !== null,
      version: PYODIDE_VERSION,
    };
  }

  function defaultInjectScript(src) {
    return new Promise((resolve, reject) => {
      if (typeof document === 'undefined') {
        reject(new Error('Pyodide can only be loaded in a browser environment.'));
        return;
      }
      const existing = document.querySelector(`script[src="${src}"]`);
      if (existing && globalThis.loadPyodide) { resolve(); return; }
      const s = document.createElement('script');
      s.src = src;
      s.async = true;
      s.onload = () => resolve();
      s.onerror = () => reject(new Error(`Failed to load Pyodide from ${src}`));
      document.head.appendChild(s);
    });
  }

  const defaultRunner = createRunner();

  return { createRunner, defaultRunner, PYODIDE_VERSION, PYODIDE_BASE_URL };
});
