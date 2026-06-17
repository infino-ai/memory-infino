// SPDX-License-Identifier: Apache-2.0
//
// LOCOMO self-evaluation for infino-backed agent memory. Independent harness —
// no third-party vector store. Per conversation: a fresh infino catalog, ingest
// every session turn as a memory (local embedder, no API key), then per QA pair
// recall → answer (LLM) → LLM-judge → score. Reports accuracy overall + per
// LOCOMO category, recall/answer latency, and token cost.
//
// Usage (LLM via env — see eval/llm.mjs):
//   OPENAI_BASE_URL=… OPENAI_API_KEY=… EVAL_MODEL=gpt-5.4 \
//     node eval/run-locomo.mjs --mode=hybrid --conversations=1 --questions=5
//   flags: --mode=hybrid|semantic|keyword  --k=12  --conversations=N  --questions=N
//
// Start with a small subset (--conversations=1 --questions=5) — the full set is
// ~10 conversations × ~200 QA and makes 2 LLM calls each.
import { InfinoMemoryStore } from "../dist/infino-store.js";
import { createEmbeddings } from "../dist/embeddings.js";
import { chat, MODEL } from "./llm.mjs";
import { mkdtempSync, mkdirSync, existsSync, writeFileSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const DATA = join(HERE, "data", "locomo10.json");
const DATA_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json";
const CATEGORY = { 1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial" };
const DIM = 384; // all-MiniLM-L6-v2

const args = Object.fromEntries(
  process.argv.slice(2).map((a) => {
    const [k, v] = a.replace(/^--/, "").split("=");
    return [k, v ?? true];
  }),
);
const MODE = String(args.mode || "hybrid"); // hybrid | semantic | keyword
const K = Number(args.k || 12);
const LIMIT_CONV = args.conversations ? Number(args.conversations) : Infinity;
const LIMIT_Q = args.questions ? Number(args.questions) : Infinity;

const embedder = createEmbeddings({ provider: "local", model: "Xenova/all-MiniLM-L6-v2" });
const embed = (t) => embedder.embed(t);

const ANSWER_SYS =
  "You answer a question using ONLY the provided memories from a past conversation. " +
  "Be concise (a few words when possible). Each memory line begins with its timestamp in parentheses; " +
  "when the answer is a date or time, resolve relative references (e.g. 'yesterday', 'last year', 'last week', " +
  "'next month', 'this morning') to an absolute date, month, or year using that timestamp. " +
  "If the answer is not in the memories, reply exactly: Not mentioned in the conversation.";
const JUDGE_SYS =
  "You are a strict grader. Given a question, the gold answer, and a predicted answer, decide whether the " +
  "prediction conveys the same answer as the gold. Reply with exactly one word: CORRECT or WRONG.";

const pct = (n, d) => (d ? ((100 * n) / d).toFixed(1) : "0.0");
const percentile = (arr, p) => {
  if (!arr.length) return 0;
  const s = [...arr].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor((p / 100) * s.length))];
};

async function loadDataset() {
  if (!existsSync(DATA)) {
    mkdirSync(dirname(DATA), { recursive: true });
    process.stderr.write("Downloading locomo10.json (once)...\n");
    const res = await fetch(DATA_URL);
    if (!res.ok) throw new Error(`dataset download failed: HTTP ${res.status}`);
    writeFileSync(DATA, await res.text());
  }
  return JSON.parse(readFileSync(DATA, "utf8"));
}

function sessionsOf(conv) {
  const out = [];
  for (let i = 1; ; i++) {
    const turns = conv[`session_${i}`];
    if (!Array.isArray(turns)) break;
    out.push({ date: conv[`session_${i}_date_time`] || "", turns });
  }
  return out;
}

async function ingest(store, conv) {
  const entries = [];
  for (const s of sessionsOf(conv)) {
    for (const turn of s.turns) {
      const t = String(turn.text ?? turn.clean_text ?? "").trim();
      if (!t) continue;
      // Prepend the session date so temporal questions are answerable.
      const text = `(${s.date}) ${turn.speaker}: ${t}`;
      entries.push({ text, vector: await embed(text), category: "fact" });
    }
  }
  if (entries.length) await store.storeMany(entries);
  return entries.length;
}

async function recall(store, question) {
  const vec = await embed(question);
  if (MODE === "semantic") return store.recallSemantic(vec, { k: K });
  if (MODE === "keyword") return store.recallKeyword(question, { k: K });
  return store.recallHybrid(question, vec, { k: K });
}

(async () => {
  process.stderr.write(`LOCOMO eval — model=${MODEL} mode=${MODE} k=${K}\n`);
  const data = await loadDataset();

  const per = {}; // category -> { n, correct }
  const recallLat = [];
  const answerLat = [];
  let total = 0;
  let correct = 0;
  let errors = 0;
  let promptTok = 0;
  let compTok = 0;
  let convN = 0;

  for (const item of data) {
    if (convN >= LIMIT_CONV) break;
    convN++;
    const conv = item.conversation;
    const store = new InfinoMemoryStore({ uri: mkdtempSync(join(tmpdir(), "locomo-")), dimensions: DIM, nCent: 1 });
    const ingested = await ingest(store, conv);
    process.stderr.write(`conv ${convN} (${item.sample_id ?? "?"}): ingested ${ingested} turns\n`);

    let qN = 0;
    for (const qa of item.qa ?? []) {
      if (qN >= LIMIT_Q) break;
      qN++;
      const question = qa.question;
      const gold = String(qa.answer ?? qa.adversarial_answer ?? "");
      const cat = CATEGORY[qa.category] ?? `cat-${qa.category ?? "?"}`;
      per[cat] ??= { n: 0, correct: 0 };
      per[cat].n++;
      total++;
      try {
        const t0 = Date.now();
        const hits = await recall(store, question);
        const t1 = Date.now();
        const mem = hits.map((h, i) => `${i + 1}. ${h.entry.text}`).join("\n");
        const ans = await chat([
          { role: "system", content: ANSWER_SYS },
          { role: "user", content: `Memories:\n${mem}\n\nQuestion: ${question}` },
        ]);
        const t2 = Date.now();
        const judge = await chat([
          { role: "system", content: JUDGE_SYS },
          { role: "user", content: `Question: ${question}\nGold answer: ${gold}\nPredicted answer: ${ans.text}` },
        ]);
        const ok = /\bcorrect\b/i.test(judge.text) && !/\bwrong\b/i.test(judge.text);
        if (ok) {
          correct++;
          per[cat].correct++;
        }
        if (args.verbose) {
          process.stderr.write(
            `  [${ok ? "✓" : "✗"}] (${cat}) Q: ${question}\n      gold: ${gold}\n      pred: ${ans.text.slice(0, 140)}\n` +
              `      top mem: ${hits[0]?.entry.text?.slice(0, 100) ?? "(none)"}\n`,
          );
        }
        recallLat.push(t1 - t0);
        answerLat.push(t2 - t1);
        promptTok += ans.promptTokens + judge.promptTokens;
        compTok += ans.completionTokens + judge.completionTokens;
      } catch (err) {
        errors++;
        process.stderr.write(`  q${qN} error: ${(err).message}\n`);
      }
    }
  }

  const summary = {
    model: MODEL,
    mode: MODE,
    k: K,
    conversations: convN,
    questions: total,
    errors,
    accuracy: {
      overall: `${pct(correct, total)}% (${correct}/${total})`,
      byCategory: Object.fromEntries(
        Object.entries(per)
          .sort()
          .map(([c, v]) => [c, `${pct(v.correct, v.n)}% (${v.correct}/${v.n})`]),
      ),
    },
    latencyMs: {
      recall_p50: percentile(recallLat, 50),
      recall_p95: percentile(recallLat, 95),
      answer_p50: percentile(answerLat, 50),
      answer_p95: percentile(answerLat, 95),
    },
    tokens: { prompt: promptTok, completion: compTok, note: "approximate — from provider usage" },
  };

  console.log(JSON.stringify(summary, null, 2));
  const outDir = join(HERE, "results");
  mkdirSync(outDir, { recursive: true });
  const out = join(outDir, `locomo-${MODE}-${convN}conv-${total}q.json`);
  writeFileSync(out, JSON.stringify(summary, null, 2));
  process.stderr.write(`\nwrote ${out}\n`);
})();
