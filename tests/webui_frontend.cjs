/* Runs the actual front end without a browser; DOM only records visible turns. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

class Element {
  constructor() {
    this.children = [];
    this.listeners = {};
    this.style = {};
    this.classList = { add() {}, remove() {}, toggle() {} };
    this.value = '';
    this.checked = false;
    this.scrollHeight = this.scrollTop = this.clientHeight = 0;
  }
  addEventListener(name, fn) { (this.listeners[name] ??= []).push(fn); }
  dispatch(name) { for (const fn of this.listeners[name] ?? []) fn({preventDefault() {}}); }
  append(child) { child.remove(); child.parent = this; this.children.push(child); }
  replaceChildren(...children) {
    for (const child of [...this.children]) child.remove();
    for (const child of children) this.append(child);
  }
  remove() {
    if (this.parent) this.parent.children = this.parent.children.filter(child => child !== this);
    this.parent = null;
  }
  focus() {}
  setAttribute() {}
}

const elements = new Map();
const element = id => {
  if (!elements.has(id)) elements.set(id, new Element());
  return elements.get(id);
};
const requests = [];
const context = vm.createContext({
  document: {getElementById: element, createElement: () => new Element()},
  AbortController, TextDecoder, performance,
  fetch: async (url, options) => {
    if (url === '/api/info') throw new Error('offline');
    const request = {payload: JSON.parse(options.body), pending: null};
    requests.push(request);
    options.signal.addEventListener('abort', () => {
      request.aborted = true;
      const error = new Error('stopped'); error.name = 'AbortError';
      request.pending?.reject(error);
    });
    return {ok: true, body: {getReader: () => ({read: () => new Promise((resolve, reject) => {
      request.pending = {resolve, reject};
      if (request.aborted) {
        const error = new Error('stopped'); error.name = 'AbortError'; reject(error);
      }
    })})}};
  },
});
vm.runInContext(fs.readFileSync(process.argv[2], 'utf8'), context);
const run = code => vm.runInContext(code, context);
const turns = () => JSON.parse(run('JSON.stringify(conversation)'));
const tick = () => new Promise(resolve => setImmediate(resolve));
async function delta(request, text) {
  assert.ok(request.pending, 'the client must be waiting for a stream read');
  const pending = request.pending;
  request.pending = null;
  pending.resolve({done: false, value: Buffer.from(`data: ${JSON.stringify({delta: text})}\n\n`)});
  await tick();
}

(async () => {
  await tick(); // /api/info failure must still leave usable defaults.
  assert.equal(run('settings().temperature'), .8);
  assert.equal(run('settings().top_k'), 0);
  assert.equal(run('settings().max_new_tokens'), 512);
  assert.equal(run('renderContent("```python\\nprint(1)\\n```")'), '<pre><code>print(1)\n</code></pre>');
  assert.equal(run('renderContent("```print(1)\\nprint(2)\\n```")'), '<pre><code>print(1)\nprint(2)\n</code></pre>');
  assert.equal(run('renderContent("```hello\\nworld\\n```")'), '<pre><code>hello\nworld\n</code></pre>');
  assert.ok(run('renderContent("```<script>bad()</script>```")').includes('&lt;script&gt;'));

  const old = run('send("old question")');
  await tick();
  await delta(requests[0], 'partial old answer');
  element('new-chat').dispatch('click');
  // Start another request before the old abort rejection runs its catch/finally.
  const fresh = run('send("fresh question")');
  await tick();
  await old;
  assert.deepEqual(turns(), [{role: 'user', content: 'fresh question'}]);
  assert.equal(run('inFlight !== null'), true, 'old finally must not clear new request');
  assert.equal(element('input').disabled, true);
  assert.deepEqual(requests[1].payload.messages, [{role: 'user', content: 'fresh question'}]);
  await delta(requests[1], 'fresh answer');
  requests[1].pending.resolve({done: true});
  await fresh;
  assert.deepEqual(turns(), [{role: 'user', content: 'fresh question'}, {role: 'assistant', content: 'fresh answer'}]);

  // Stop before any delta: remove both rejected turns from state and DOM.
  const before = element('messages').children.length;
  const stopped = run('send("cancel this")');
  await tick();
  element('stop').dispatch('click');
  await stopped;
  assert.equal(element('messages').children.length, before);
  assert.equal(turns().length, 2);

  // Stop after a delta retains a coherent user/assistant pair.
  const partial = run('send("keep partial")');
  await tick();
  await delta(requests[3], 'kept answer');
  element('stop').dispatch('click');
  await partial;
  assert.deepEqual(turns().slice(-2), [{role: 'user', content: 'keep partial'}, {role: 'assistant', content: 'kept answer'}]);
  console.log('frontend race, cancellation, defaults, and rendering regressions passed');
})().catch(error => { console.error(error); process.exitCode = 1; });
