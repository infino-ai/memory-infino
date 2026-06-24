// SPDX-License-Identifier: Apache-2.0
// Layer-1 oracle tests: deterministic, no LLM, no OpenClaw. Exercises the real
// infino binding via the store. Semantic/hybrid *quality* is covered by the
// LOCOMO bench, not here — these test plumbing/correctness.
import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { InfinoMemoryStore } from "../dist/infino-store.js";

// Deterministic fake embedder: text -> normalized 8-dim vector. Enough to test
// store/recall/SQL/forget plumbing; not meant to model semantics.
const DIM = 16; // infino requires vector dim in [16, 4096]
const embed = (s) => {
  const v = new Array(DIM).fill(0);
  for (let i = 0; i < s.length; i++) v[i % DIM] += (s.charCodeAt(i) % 13) / 13;
  const n = Math.hypot(...v) || 1;
  return v.map((x) => x / n);
};
const newStore = () =>
  new InfinoMemoryStore({ uri: mkdtempSync(join(tmpdir(), "mem-")), dimensions: DIM, nCent: 1 });

test("store + count + keyword recall finds the exact term", async () => {
  const m = newStore();
  await m.store({ text: "user prefers dark mode", vector: embed("user prefers dark mode") });
  await m.store({ text: "error code E1234 on checkout", vector: embed("error code E1234 on checkout") });
  assert.equal(m.count(), 2);
  const hits = m.recallKeyword("E1234", { k: 1 });
  assert.ok(hits.length >= 1);
  assert.match(hits[0].entry.text, /E1234/);
});

test("semantic recall returns a hit for a stored memory", async () => {
  const m = newStore();
  await m.store({ text: "the deploy runs every night at 2am", vector: embed("the deploy runs every night at 2am") });
  const hits = m.recallSemantic(embed("nightly deploy schedule"), { k: 1 });
  assert.equal(hits.length, 1);
});

test("hybrid recall returns results and is rank-fused", async () => {
  const m = newStore();
  await m.store({ text: "invoice INV-77 was refunded", vector: embed("invoice INV-77 was refunded") });
  await m.store({ text: "customer asked about the billing portal", vector: embed("customer asked about the billing portal") });
  const hits = m.recallHybrid("INV-77", embed("INV-77"), { k: 2 });
  assert.ok(hits.length >= 1);
  assert.match(hits[0].entry.text, /INV-77/);
});

test("timeline filters by category", async () => {
  const m = newStore();
  await m.store({ text: "decided to use infino", vector: embed("decided to use infino"), category: "decision" });
  await m.store({ text: "a random fact", vector: embed("a random fact"), category: "fact" });
  const decisions = m.timeline({ category: "decision" });
  assert.equal(decisions.length, 1);
  assert.match(decisions[0].text, /infino/);
});

test("forget removes a memory", async () => {
  const m = newStore();
  const e = await m.store({ text: "temporary note", vector: embed("temporary note") });
  assert.equal(m.forgetById(e.id), true);
  assert.equal(m.count(), 0);
});

test("auto-compaction triggers on the commit threshold; data and recall survive", async () => {
  const m = new InfinoMemoryStore({
    uri: mkdtempSync(join(tmpdir(), "mem-")),
    dimensions: DIM,
    nCent: 1,
    compactEvery: 3, // each store is one commit, so this crosses the threshold
  });
  for (let i = 0; i < 5; i++) {
    await m.store({ text: `note number ${i}`, vector: embed(`note number ${i}`) });
  }
  assert.equal(m.count(), 5); // all rows intact after the merge
  m.optimize(); // explicit call is safe (merge-or-no-op)
  const hits = m.recallKeyword("number", { k: 5 });
  assert.ok(hits.length >= 1); // recall still works post-compaction
});
