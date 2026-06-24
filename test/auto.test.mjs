// SPDX-License-Identifier: Apache-2.0
// Tests for auto-recall / auto-capture: pure helpers + the full hook flow
// (fake OpenClaw api + real InfinoMemoryStore + deterministic embedder).
import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { InfinoMemoryStore } from "../dist/infino-store.js";
import {
  extractUserText,
  latestUserText,
  shouldCapture,
  detectCategory,
  formatMemoriesSection,
  resolveStartIndex,
  fingerprint,
  installAutoMemory,
} from "../dist/auto.js";

const DIM = 16;
const embed = async (s) => {
  const v = new Array(DIM).fill(0);
  for (let i = 0; i < s.length; i++) v[i % DIM] += (s.charCodeAt(i) % 13) / 13;
  const n = Math.hypot(...v) || 1;
  return v.map((x) => x / n);
};

test("extractUserText handles string + block content, ignores non-user", () => {
  assert.deepEqual(extractUserText({ role: "user", content: "hi" }), ["hi"]);
  assert.deepEqual(
    extractUserText({ role: "user", content: [{ type: "text", text: "a" }, { type: "image" }] }),
    ["a"],
  );
  assert.deepEqual(extractUserText({ role: "assistant", content: "nope" }), []);
});

test("latestUserText returns the most recent user text", () => {
  const msgs = [
    { role: "user", content: "first" },
    { role: "assistant", content: "reply" },
    { role: "user", content: "second" },
  ];
  assert.equal(latestUserText(msgs), "second");
});

test("shouldCapture: triggers in, noise out", () => {
  assert.equal(shouldCapture("I prefer dark mode over light"), true);
  assert.equal(shouldCapture("we decided to use Infino for memory"), true);
  assert.equal(shouldCapture("hello there how are you today"), false); // no trigger
  assert.equal(shouldCapture("ok"), false); // too short
  assert.equal(shouldCapture("<relevant-memories>\n1. [fact] x\n</relevant-memories>"), false);
});

test("detectCategory classifies", () => {
  assert.equal(detectCategory("I prefer tabs"), "preference");
  assert.equal(detectCategory("we decided to ship Friday"), "decision");
  assert.equal(detectCategory("contact me at a@b.com"), "entity");
  assert.equal(detectCategory("the sky is blue"), "fact");
});

test("formatMemoriesSection wraps hits in an untrusted block; empty -> ''", () => {
  assert.equal(formatMemoriesSection([]), "");
  const s = formatMemoriesSection([{ score: 1, entry: { category: "preference", text: "dark mode" } }]);
  assert.match(s, /<relevant-memories>/);
  assert.match(s, /untrusted/);
  assert.match(s, /dark mode/);
});

test("resolveStartIndex: no cursor -> 0; fingerprint match resumes after it", () => {
  const msgs = [{ role: "user", content: "a" }, { role: "user", content: "b" }];
  assert.equal(resolveStartIndex(msgs, undefined), 0);
  assert.equal(resolveStartIndex(msgs, { nextIndex: 1, fingerprint: fingerprint(msgs[0]) }), 1);
});

test("end-to-end: agent_end captures, before_prompt_build injects, cursor de-dups", async () => {
  const store = new InfinoMemoryStore({
    uri: mkdtempSync(join(tmpdir(), "auto-")),
    dimensions: DIM,
    nCent: 1,
  });
  const handlers = {};
  const api = { on: (e, h) => (handlers[e] = h), logger: { info() {}, warn() {} } };
  installAutoMemory({ api, store, embed, autoRecall: true, autoCapture: true, recallK: 8, recallMaxChars: 1000 });

  // auto-capture: a salient user statement is stored
  const turn = {
    success: true,
    messages: [{ role: "user", content: "I prefer dark mode and using tabs" }],
  };
  await handlers["agent_end"](turn, { sessionKey: "s1" });
  assert.equal(store.count(), 1);

  // auto-recall: a later prompt injects the memory as a context section
  const res = await handlers["before_prompt_build"]({
    prompt: "remind me about dark mode",
    messages: [{ role: "user", content: "remind me about dark mode" }],
  });
  assert.ok(res && typeof res.prependContext === "string");
  assert.match(res.prependContext, /dark mode/);

  // cursor: replaying the same turn does not double-capture
  await handlers["agent_end"](turn, { sessionKey: "s1" });
  assert.equal(store.count(), 1);
});

test("auto-capture skips non-salient turns", async () => {
  const store = new InfinoMemoryStore({
    uri: mkdtempSync(join(tmpdir(), "auto-")),
    dimensions: DIM,
    nCent: 1,
  });
  const handlers = {};
  const api = { on: (e, h) => (handlers[e] = h), logger: { info() {}, warn() {} } };
  installAutoMemory({ api, store, embed, autoRecall: false, autoCapture: true, recallK: 8, recallMaxChars: 1000 });
  await handlers["agent_end"](
    { success: true, messages: [{ role: "user", content: "hello there how are you today" }] },
    { sessionKey: "s2" },
  );
  assert.equal(store.count(), 0);
});
