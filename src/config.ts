// SPDX-License-Identifier: Apache-2.0
import { buildJsonPluginConfigSchema } from "openclaw/plugin-sdk/plugin-entry";
import type { InfinoStoreConfig } from "./infino-store.js";

export const MEMORY_CATEGORIES = ["preference", "fact", "decision", "entity", "other"] as const;
export type MemoryCategory = (typeof MEMORY_CATEGORIES)[number];

export const DEFAULTS = {
  dbPath: "memory/infino",
  recallK: 8,
  recallMaxChars: 1000,
  nCent: 1,
} as const;

// Built-in dims for the common OpenAI models; other models must set dimensions.
const KNOWN_DIMS: Record<string, number> = {
  "text-embedding-3-small": 1536,
  "text-embedding-3-large": 3072,
};

// Schema passed to the host (validation + config UI): the standard embedding
// and storage options plus infino's `nCent`. Built from JSON Schema (mirrors
// openclaw.plugin.json's configSchema).
export const memoryConfigSchema = buildJsonPluginConfigSchema({
  type: "object",
  additionalProperties: false,
  properties: {
    embedding: {
      type: "object",
      minProperties: 1,
      additionalProperties: false,
      properties: {
        provider: { type: "string" },
        model: { type: "string" },
        dimensions: { type: "integer", minimum: 1 },
        apiKey: { type: "string" },
        baseUrl: { type: "string" },
      },
    },
    dbPath: { type: "string" },
    nCent: { type: "integer", minimum: 1 },
    autoRecall: { type: "boolean" },
    autoCapture: { type: "boolean" },
    recallK: { type: "number" },
    recallMaxChars: { type: "number", minimum: 100, maximum: 10000 },
    storageOptions: { type: "object", additionalProperties: { type: "string" } },
  },
});

export interface MemoryConfig {
  embedding: { provider?: string; model: string; dimensions?: number; apiKey?: string; baseUrl?: string };
  dbPath?: string;
  nCent?: number;
  autoRecall?: boolean;
  autoCapture?: boolean;
  recallK?: number;
  recallMaxChars?: number;
  storageOptions?: Record<string, string>;
}

export function parseConfig(raw: unknown): MemoryConfig {
  const c = (raw ?? {}) as Record<string, any>;
  const emb = (c.embedding ?? {}) as Record<string, any>;
  const provider = (emb.provider ?? "local") as string;
  // Local provider defaults to a known model; remote providers must name one.
  const model = emb.model ?? (provider.toLowerCase() === "local" ? "Xenova/all-MiniLM-L6-v2" : undefined);
  if (!model) throw new Error(`memory-infino: config.embedding.model is required for provider "${provider}"`);
  return {
    embedding: {
      provider,
      model,
      dimensions: emb.dimensions,
      apiKey: emb.apiKey,
      baseUrl: emb.baseUrl,
    },
    dbPath: c.dbPath,
    nCent: c.nCent,
    autoRecall: c.autoRecall,
    autoCapture: c.autoCapture,
    recallK: c.recallK,
    recallMaxChars: c.recallMaxChars,
    storageOptions: c.storageOptions,
  };
}

export function resolveDimensions(cfg: MemoryConfig): number {
  if (typeof cfg.embedding.dimensions === "number") return cfg.embedding.dimensions;
  if ((cfg.embedding.provider ?? "local").toLowerCase() === "local") return 384; // all-MiniLM-L6-v2
  const d = KNOWN_DIMS[cfg.embedding.model];
  if (!d) throw new Error(`memory-infino: set config.embedding.dimensions for model "${cfg.embedding.model}"`);
  return d;
}

export function resolveStoreConfig(api: { resolvePath(p: string): string }, cfg: MemoryConfig): InfinoStoreConfig {
  const dbPath = cfg.dbPath ?? DEFAULTS.dbPath;
  const uri = /^[a-z][a-z0-9+.-]*:\/\//.test(dbPath) ? dbPath : api.resolvePath(dbPath);
  return {
    uri,
    dimensions: resolveDimensions(cfg),
    nCent: cfg.nCent ?? DEFAULTS.nCent,
    connectOptions: mapStorage(cfg.storageOptions),
  };
}

// storageOptions keys -> infino ConnectOptions (S3-compatible stores).
function mapStorage(o?: Record<string, string>): InfinoStoreConfig["connectOptions"] {
  if (!o || (!o.endpoint && !o.access_key)) return undefined;
  return { endpoint: o.endpoint, region: o.region ?? "auto", accessKey: o.access_key, secretKey: o.secret_key };
}
