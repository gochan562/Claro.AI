'use strict';
// Notebook cell behavior tests: insertion order, auto-scroll, focus,
// divider rendering, and preserved execution paths.
//
// Drives the REAL public/dashboard-app.js in jsdom with stubbed
// browser-only dependencies (CodeMirror, marked, DOMPurify, Pyodide,
// fetch). No network, no GPU, no real Python.

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const { JSDOM } = require('jsdom');

const APP_SRC = fs.readFileSync(path.join(__dirname, '..', 'public', 'dashboard-app.js'), 'utf8');

function makeWindow() {
  const dom = new JSDOM(
    `<!DOCTYPE html><html><body>` +
    `<div class="notebook-toolbar"></div>` +
    `<div id="notebook-cells"></div>` +
    `<div id="gpu-terminal"></div>` +
    `</body></html>`,
    { url: 'http://localhost:5000/', runScripts: 'dangerously' }
  );
  const w = dom.window;

  // --- stub CodeMirror: functional fake recording focus/value ---
  const editors = [];
  w.CodeMirror = function (host, opts) {
    const ed = {
      _value: opts.value || '',
      _focused: false,
      _handlers: {},
      getValue: () => ed._value,
      setValue: (v) => { ed._value = v; },
      on: (ev, fn) => { ed._handlers[ev] = fn; },
      focus: () => { ed._focused = true; },
      refresh: () => {},
      getWrapperElement: () => w.document.createElement('div'),
    };
    const el = w.document.createElement('div');
    el.className = 'CodeMirror';
    host.appendChild(el);
    editors.push(ed);
    return ed;
  };

  w.marked = { parse: (s) => `<p>${s}</p>` };
  w.DOMPurify = { sanitize: (s) => s };
  w.PyodideLocal = { run: async () => ({ ok: true, stdout: '3', stderr: '', error: null }) };
  w.fetch = async () => { throw new Error('no network in tests'); };
  w.prompt = () => null;
  w.confirm = () => false;

  const script = w.document.createElement('script');
  script.textContent = APP_SRC;
  w.document.body.appendChild(script);

  // Fake layout at the prototype level (jsdom has no layout engine, and
  // re-renders recreate elements, so per-element stubs would not survive).
  Object.defineProperty(w.HTMLElement.prototype, 'offsetTop', {
    get() {
      if (!this.classList || !this.classList.contains('notebook-cell')) return 0;
      const cells = Array.from(this.parentNode.querySelectorAll('.notebook-cell'));
      return cells.indexOf(this) * 400;
    },
    configurable: true,
  });
  Object.defineProperty(w.HTMLElement.prototype, 'clientHeight', {
    get() { return this.id === 'notebook-cells' ? 1000 : 0; },
    configurable: true,
  });

  for (const fn of ['createNotebook', 'setActiveNotebook', 'addCell', 'insertCellAfter',
    'deleteCell', 'runCell', 'updateCellContent', 'getActiveNotebook',
    'scrollToCell', 'focusCellEditor']) {
    assert.strictEqual(typeof w[fn], 'function', `dashboard must expose global ${fn}()`);
  }
  return { dom, w, editors };
}

function newNotebook(w) {
  const id = w.createNotebook('Test');
  w.setActiveNotebook(id);
  return id;
}

// Fake layout: jsdom has no layout engine, so assign offsets manually.
// (Kept for API compatibility; real geometry comes from the prototype
// fakes installed in makeWindow, which survive re-renders.)
function layoutCells(w, cellHeight = 400, viewportHeight = 1000) {
  const container = w.document.getElementById('notebook-cells');
  const cells = Array.from(container.querySelectorAll('.notebook-cell'));
  return { container, cells };
}

function stubScroll(container) {
  const calls = [];
  container.scrollTo = (opts) => { calls.push(opts); container.scrollTop = opts.top; };
  return calls;
}

function cellOrder(w) {
  // Array.from re-homes the array into this realm (jsdom arrays fail
  // deepStrictEqual across realms).
  return Array.from(w.getActiveNotebook().cells, (c) => `${c.id}:${c.type}`);
}

let passed = 0, failed = 0;
function test(name, fn) {
  return Promise.resolve()
    .then(fn)
    .then(() => { console.log(`✓ ${name}`); passed++; })
    .catch((e) => { console.error(`✗ ${name}: ${e.message}`); failed++; });
}

async function main() {
  console.log('=== Notebook Cell Behavior Tests ===\n');

  // 1. Bottom add still works: order, ids, rendering.
  await test('1. addCell appends at the bottom with stable ids', () => {
    const { w } = makeWindow();
    newNotebook(w); // seeds code cell id 1
    w.addCell('code');
    w.addCell('markdown');
    assert.deepStrictEqual(cellOrder(w), ['1:code', '2:code', '3:markdown']);
    assert.strictEqual(w.document.querySelectorAll('.notebook-cell').length, 3);
    assert(w.document.getElementById('cell-3').querySelector('.markdown-render'),
      'markdown cell must render its preview div');
  });

  // 2. Insert between cells at the exact position.
  await test('2. insertCellAfter inserts between cells', () => {
    const { w } = makeWindow();
    newNotebook(w);
    w.addCell('code');    // id 2
    w.addCell('code');    // id 3
    w.insertCellAfter(1, 'markdown'); // id 4 between 1 and 2
    assert.deepStrictEqual(cellOrder(w), ['1:code', '4:markdown', '2:code', '3:code']);
    const ids = Array.from(w.document.querySelectorAll('.notebook-cell'), (el) => el.id);
    assert.deepStrictEqual(ids, ['cell-1', 'cell-4', 'cell-2', 'cell-3']);
  });

  // 3. Unknown anchor falls back to appending (never loses the cell).
  await test('3. insertCellAfter with unknown anchor appends', () => {
    const { w } = makeWindow();
    newNotebook(w);
    w.insertCellAfter(999, 'code');
    assert.deepStrictEqual(cellOrder(w), ['1:code', '2:code']);
  });

  // 4. Divider affordance exists after every cell and really inserts.
  await test('4. divider buttons insert at that position', () => {
    const { w } = makeWindow();
    newNotebook(w);
    w.addCell('code'); // id 2
    const dividers = [...w.document.querySelectorAll('.cell-insert-divider')];
    assert.strictEqual(dividers.length, 2, 'one divider per cell (incl. after last)');
    assert.strictEqual(dividers[0].getAttribute('data-after-cell'), '1');
    const btns = dividers[0].querySelectorAll('.cell-insert-btn');
    assert.strictEqual(btns.length, 2, '+ Code and + Markdown buttons');
    btns[1].click(); // "+ Markdown" after cell 1
    assert.deepStrictEqual(cellOrder(w), ['1:code', '3:markdown', '2:code']);
  });

  // 5. New cell auto-scrolls into the lower-middle of the container.
  await test('5. new cell scrolls into view (container, not window)', () => {
    const { w } = makeWindow();
    newNotebook(w);
    w.addCell('code');
    w.addCell('code');
    const { container } = layoutCells(w, 400, 1000);
    const calls = stubScroll(container);
    w.addCell('code'); // id 4 at index 3 → 3*400 - 1000*0.35 = 850
    assert.strictEqual(calls.length, 1, 'exactly one scroll on add');
    assert.strictEqual(calls[0].top, 850);
    assert(calls[0].top > 0, 'must actually move (not reveal 1px)');
  });

  // 6. Inserted cell scrolls + focuses its editor.
  await test('6. inserted cell scrolls and focuses', () => {
    const { w, editors } = makeWindow();
    newNotebook(w);
    w.addCell('code');
    w.addCell('code');
    const { container } = layoutCells(w, 400, 1000);
    const calls = stubScroll(container);
    editors.forEach((e) => { e._focused = false; });
    w.insertCellAfter(1, 'code'); // id 4 at index 1 → 1*400-350 = 50
    assert.strictEqual(calls.length, 1);
    assert.strictEqual(calls[0].top, 50);
    const focused = editors.filter((e) => e._focused);
    assert.strictEqual(focused.length, 1, 'exactly the new editor is focused');
  });

  // 7. No scroll jump on unrelated updates.
  await test('7. unrelated updates do not move scroll', () => {
    const { w } = makeWindow();
    newNotebook(w);
    w.addCell('code');
    const { container } = layoutCells(w);
    const calls = stubScroll(container);
    w.updateCellContent(1, 'x = 1');
    w.deleteCell(2);
    assert.strictEqual(calls.length, 0, 'no scroll on content update/delete');
    assert.deepStrictEqual(cellOrder(w), ['1:code']);
  });

  // 8. Code cell still executes (stubbed Pyodide) and renders output.
  await test('8. code cell executes and renders output', async () => {
    const { w } = makeWindow();
    newNotebook(w);
    await w.runCell(1);
    const out = w.document.querySelector('#cell-1 .cell-output pre');
    assert(out && out.textContent === '3', `expected output 3, got ${out && out.textContent}`);
  });

  // 9. Markdown cell still renders.
  await test('9. markdown cell renders', async () => {
    const { w } = makeWindow();
    newNotebook(w);
    w.addCell('markdown');
    w.updateCellContent(2, '# Hi');
    await w.runCell(2);
    const cell = w.getActiveNotebook().cells.find((c) => c.id === 2);
    assert.strictEqual(cell.outputFormat, 'markdown');
    assert(cell.output.includes('<p>'), 'rendered markdown present');
  });

  // 10. focusCellEditor is a safe no-op without an editor.
  await test('10. focusCellEditor safe without editor', () => {
    const { w } = makeWindow();
    newNotebook(w);
    assert.strictEqual(w.focusCellEditor(99999), false);
  });

  console.log(`\n=== Results: ${passed} passed, ${failed} failed ===`);
  if (failed) process.exit(1);
}

main().catch((e) => { console.error('TEST FAILED:', e); process.exit(1); });
