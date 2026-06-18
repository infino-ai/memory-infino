// SPDX-License-Identifier: Apache-2.0
// Self-contained embedder built from the plugin's embedding config. Supports
// OpenAI-compatible endpoints and Ollama via fetch. (Reusing the host's
// configured provider auth profiles is a future enhancement; this covers OpenAI
// API keys and local Ollama, which is what the demo + benchmark need.)
import type { MemoryConfig } from "./config.js";

export interface Embedder {
  embed(text: string): Promise<number[]>;
}

export function createEmbeddings(embedding: MemoryConfig["embedding"]): Embedder {
  const provider = (embedding.provider ?? "local").toLowerCase();
  if (provider === "local") return localEmbedder(embedding);
  if (provider === "ollama") return ollamaEmbedder(embedding);
  return openAiCompatibleEmbedder(embedding);
}

// Local, in-process embedder via Hugging Face transformers.js: no API key, no
// external service. The model downloads once and is cached. This is the default
// (zero-setup) path; `all-MiniLM-L6-v2` is 384-dim.
function localEmbedder(e: MemoryConfig["embedding"]): Embedder {
  const model = e.model || "Xenova/all-MiniLM-L6-v2";
  let pipe: Promise<(t: string, o: object) => Promise<{ data: ArrayLike<number> }>> | null = null;
  const getPipe = () => {
    if (!pipe) {
      pipe = (async () => {
        const { pipeline } = await import("@huggingface/transformers");
        return (await pipeline("feature-extraction", model)) as never;
      })();
    }
    return pipe;
  };
  return {
    async embed(text: string): Promise<number[]> {
      const extractor = await getPipe();
      const out = await extractor(text, { pooling: "mean", normalize: true }); // sentence embedding
      return Array.from(out.data, Number);
    },
  };
}

function ollamaEmbedder(e: MemoryConfig["embedding"]): Embedder {
  const base = (e.baseUrl ?? "http://127.0.0.1:11434").replace(/\/+$/, "");
  return {
    async embed(text: string): Promise<number[]> {
      const res = await fetch(`${base}/api/embeddings`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: e.model, prompt: text }),
      });
      if (!res.ok) throw new Error(`ollama embeddings ${res.status}: ${await res.text()}`);
      const json = (await res.json()) as { embedding?: number[] };
      if (!Array.isArray(json.embedding)) throw new Error("ollama: response missing 'embedding'");
      return json.embedding;
    },
  };
}

function openAiCompatibleEmbedder(e: MemoryConfig["embedding"]): Embedder {
  const base = (e.baseUrl ?? "https://api.openai.com/v1").replace(/\/+$/, "");
  return {
    async embed(text: string): Promise<number[]> {
      const res = await fetch(`${base}/embeddings`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          // Send both auth headers: OpenAI uses the Bearer token, Azure OpenAI
          // authenticates the embeddings endpoint with the `api-key` header
          // (Bearer there is reserved for AAD tokens). Sending both works for
          // either provider.
          ...(e.apiKey ? { Authorization: `Bearer ${e.apiKey}`, "api-key": e.apiKey } : {}),
        },
        body: JSON.stringify({
          model: e.model,
          input: text,
          ...(typeof e.dimensions === "number" ? { dimensions: e.dimensions } : {}),
        }),
      });
      if (!res.ok) throw new Error(`embeddings ${res.status}: ${await res.text()}`);
      const json = (await res.json()) as { data?: Array<{ embedding?: number[] }> };
      const vec = json.data?.[0]?.embedding;
      if (!Array.isArray(vec)) throw new Error("embeddings: response missing a vector");
      return vec;
    },
  };
}
