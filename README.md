# memory-infino

An [OpenClaw](https://docs.openclaw.ai) memory plugin backed by [Infino](https://github.com/infino-ai/infino) — **hybrid (BM25 + vector) recall** on **object storage**, in one engine.

> Status: early. Implements OpenClaw's memory-plugin contract — the three tools (`memory_recall` / `memory_store` / `memory_forget`) plus **auto-recall** (relevant memories injected into context each turn) and **auto-capture** (salient user statements stored automatically). Built on the published [`@infino-ai/infino`](https://www.npmjs.com/package/@infino-ai/infino) Node binding.

## Why infino for agent memory

- **Hybrid recall in one engine** — `memory_recall` fuses BM25 + vector with Reciprocal Rank Fusion. No separate keyword index, no rerank service.
- **Seamless by default** — **auto-recall** injects relevant memories into context each turn and **auto-capture** stores salient statements automatically; the explicit tools stay for manual control. Both are configurable.
- **Object-storage-native** — memory lives on a local path or S3/Azure; no daemon.
- **SQL over memory** — the underlying store also supports structured/temporal queries (used internally; a future tool surface).

## Tools

| Tool | What it does |
| --- | --- |
| `memory_recall` | Hybrid (BM25 + vector) search over long-term memory |
| `memory_store` | Save a memory (embedded on write) |
| `memory_forget` | Delete by id, or find-and-delete by query |

## Install

Install the plugin into OpenClaw and select it:

```sh
openclaw plugins install @infino-ai/memory-infino
openclaw plugins enable memory-infino          # plugin id is "memory-infino"
# set plugins.slots.memory = "memory-infino" + an embedder (see Configure)
openclaw                                        # restart the gateway
```

> Note the two names: the **npm package** is `@infino-ai/memory-infino`; the
> **OpenClaw plugin id** (used for `plugins.slots.memory` and `plugins.entries`)
> is `memory-infino`.

## Configure

Select the plugin for the memory slot and give it an embedder + a path:

```json5
{
  plugins: {
    slots: { memory: "memory-infino" },
    entries: {
      "memory-infino": {
        enabled: true,
        config: {
          // Default: local in-process embedder — no API key, no service. all-MiniLM-L6-v2 = 384-dim.
          embedding: { provider: "local", model: "Xenova/all-MiniLM-L6-v2", dimensions: 384 },
          // or Ollama: { provider: "ollama", baseUrl: "http://127.0.0.1:11434", model: "mxbai-embed-large", dimensions: 1024 }
          // or OpenAI: { provider: "openai", apiKey: "${OPENAI_API_KEY}", model: "text-embedding-3-small" }
          dbPath: "memory/infino",          // local path, or s3://… / az://… for object storage
          // storageOptions: { aws_access_key_id: "${AWS_ACCESS_KEY_ID}", aws_secret_access_key: "${AWS_SECRET_ACCESS_KEY}", aws_region: "us-east-1" }
        },
      },
    },
  },
}
```

| Key | Meaning |
| --- | --- |
| `embedding` | provider/model/dimensions (+ apiKey/baseUrl). The plugin embeds text via this provider (infino is bring-your-own-vectors), so `dimensions` must match the model (infino requires 16–4096). Default provider is `local`. |
| `dbPath` | local path or `s3://`/`az://` URI (default `memory/infino`) |
| `nCent` | infino IVF centroid count; `1` = exact (right for memory-scale stores) |
| `compactEvery` | auto-merge accumulated superfiles after this many stores (each store is one commit); keeps recall fast as memory grows. `0` disables. Default `128`. |
| `autoRecall` | inject relevant memories into context automatically each turn (no tool call). Default `true`. |
| `autoCapture` | store salient user statements (preferences / decisions / facts) automatically. Default `true`. |
| `captureMaxChars` / `captureMaxPerTurn` | skip auto-capturing messages longer than this (default 2000) / cap memories stored per turn (default 3) |
| `storageOptions` | cloud credentials keyed by `object_store` config strings — S3 `aws_access_key_id`/`aws_secret_access_key`/`aws_region` (+ `aws_endpoint` for R2/MinIO), Azure `azure_storage_account_name`/`azure_storage_account_key`; omit for ambient cloud identity |
| `recallK` / `recallMaxChars` | results per recall (default 8) / max query chars embedded (default 1000) |

Supported embedders today: **`local`** (the default — in-process Hugging Face transformers.js, `all-MiniLM-L6-v2`, 384-dim, no API key and no external service), **Ollama** (`provider: "ollama"`), and **OpenAI-compatible** (`provider: "openai"` + `apiKey`/`baseUrl`). Reusing OpenClaw's configured provider auth profiles is a future enhancement.

## Local testing — no publish required

**Layer 0 — the store, without OpenClaw (fastest signal):**
```sh
npm install      # installs deps incl. @infino-ai/infino from public npm
npm test         # builds + runs deterministic oracle tests on real infino
```

**Layer 1 — inside OpenClaw, via local link:**
```sh
npm run build
openclaw plugins install --link "$(pwd)"   # references this dir; no copy
openclaw plugins enable memory-infino
# set plugins.slots.memory = "memory-infino" and an embedder in config (above)
openclaw                                     # restart the gateway; edits apply on restart
```

## Scope

Hybrid recall + auto-recall / auto-capture + object storage, via the three standard memory tools. Deliberately out of scope for now: cross-encoder reranking, multi-scope (per-user/session) isolation, and a management CLI.

## Layout

```
src/
├── infino-store.ts   # store on infino: hybrid recall (RRF) / SQL / forget / optimize  (oracle-tested)
├── auto.ts           # auto-recall + auto-capture lifecycle hooks (+ pure helpers)
├── config.ts         # config schema + OpenClaw→infino mapping
├── embeddings.ts     # local (default, transformers.js) + Ollama + OpenAI-compatible embedders
└── index.ts          # OpenClaw plugin entry: the three tools + installs the auto hooks
test/
├── infino-store.test.mjs   # store oracle tests
└── auto.test.mjs           # auto-recall/capture helpers + end-to-end hook flow
openclaw.plugin.json  # plugin manifest (kind: memory, tools, config schema)
```
