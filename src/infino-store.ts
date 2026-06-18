// SPDX-License-Identifier: Apache-2.0
// Infino-backed long-term memory: hybrid (BM25+vector) recall + SQL over memory,
// on object storage. Self-contained on the published `infino` Node binding.
import { connect, IndexSpec, type Connection, type Table, type ConnectOptions } from "infino";
import { randomUUID } from "node:crypto";

export type MemoryCategory = "preference" | "decision" | "entity" | "fact" | "other";

export interface MemoryEntry {
  id: string;
  text: string;
  vector: number[];
  importance: number;
  category: MemoryCategory;
  createdAt: number; // ms since epoch
}
export interface MemoryHit {
  entry: MemoryEntry;
  score: number; // higher = better
}

export interface RecallOptions {
  k?: number;
  candidates?: number; // per-modality over-fetch before fusion/filter
  category?: MemoryCategory; // interim client-side scope (-> #173 VectorFilter later)
  since?: number;
  until?: number;
}
export interface InfinoStoreConfig {
  uri: string; // local path or s3://|gs://|az://
  dimensions: number; // must match the embedder OpenClaw is configured with
  nCent?: number; // IVF centroids; 1 = exact (good for memory-scale stores)
  table?: string;
  connectOptions?: ConnectOptions;
}

type Row = Record<string, unknown>;

const DEFAULT_K = 8;
const sqlStr = (s: string) => s.replace(/'/g, "''"); // single-quote escape

export class InfinoMemoryStore {
  private db: Connection;
  private handle: Table | null = null;
  private readonly table: string;
  private readonly projection = ["id", "text", "importance", "category", "created_at", "score"];

  constructor(private cfg: InfinoStoreConfig) {
    this.db = connect(cfg.uri, cfg.connectOptions);
    this.table = cfg.table ?? "memories";
  }

  // Open-or-create — WRITES ONLY. The create path must be followed by an append
  // (store does this) to commit a manifest; a created-but-never-appended table
  // has no manifest and can never be reopened. So reads must NOT use this.
  private ensureTable(): Table {
    if (this.handle) return this.handle;
    try {
      this.handle = this.db.openTable(this.table);
    } catch {
      const idx = new IndexSpec() // builder returns a new spec — must chain
        .fts("text")
        .vector("vector", this.cfg.dimensions, this.cfg.nCent ?? 1, "cosine");
      this.handle = this.db.createTable(
        this.table,
        {
          id: "large_utf8",
          text: "large_utf8",
          vector: { vector: this.cfg.dimensions },
          importance: "float64",
          category: "large_utf8",
          created_at: "float64",
        },
        idx,
      );
    }
    return this.handle;
  }

  // Read-only open: null when the table doesn't exist yet (no memories stored).
  // NEVER creates — creating on a read leaves a manifest-less, unopenable table.
  private readTable(): Table | null {
    if (this.handle) return this.handle;
    try {
      this.handle = this.db.openTable(this.table);
      return this.handle;
    } catch {
      return null;
    }
  }

  // ---------- write ----------
  async store(input: {
    text: string;
    vector: number[];
    importance?: number;
    category?: MemoryCategory;
    id?: string;
  }): Promise<MemoryEntry> {
    const entry = this.makeEntry(input);
    this.ensureTable().append([this.dbRow(entry)] as unknown as Row[]); // one append = one commit
    return entry;
  }

  /** Batch ingest in a single commit (corpus build / benchmark). */
  async storeMany(inputs: Array<Parameters<InfinoMemoryStore["store"]>[0]>): Promise<number> {
    const entries = inputs.map((i) => this.makeEntry(i));
    this.ensureTable().append(entries.map((e) => this.dbRow(e)) as unknown as Row[]);
    return entries.length;
  }

  private makeEntry(i: {
    text: string;
    vector: number[];
    importance?: number;
    category?: MemoryCategory;
    id?: string;
  }): MemoryEntry {
    return {
      id: i.id ?? randomUUID(),
      text: i.text,
      vector: i.vector,
      importance: i.importance ?? 0.5,
      category: i.category ?? "other",
      createdAt: Date.now(),
    };
  }

  /** Map the JS entry (camelCase) to the DB row (snake_case columns so SQL
   *  identifiers stay unquoted — infino lowercases unquoted identifiers). */
  private dbRow(e: MemoryEntry): Row {
    return {
      id: e.id,
      text: e.text,
      vector: e.vector,
      importance: e.importance,
      category: e.category,
      created_at: e.createdAt,
    };
  }

  // ---------- recall ----------
  /** Pure semantic recall (vector kNN). */
  recallSemantic(vector: number[], opts: RecallOptions = {}): MemoryHit[] {
    const k = opts.k ?? DEFAULT_K;
    const t = this.readTable();
    if (!t) return [];
    const rows = t.vectorSearch("vector", vector, this.fetchN(opts, k), {
      projection: this.projection,
    }) as Row[];
    // infino returns a distance (lower = closer) -> map to higher-is-better.
    const hits = rows.map((r) => this.rowToHit(r, 1 / (1 + Number(r.score))));
    return this.scope(hits, opts).slice(0, k);
  }

  /** Pure keyword recall (BM25). */
  recallKeyword(query: string, opts: RecallOptions = {}): MemoryHit[] {
    const k = opts.k ?? DEFAULT_K;
    const t = this.readTable();
    if (!t) return [];
    const rows = t.bm25Search("text", query, this.fetchN(opts, k), {
      projection: this.projection,
    }) as Row[];
    const hits = rows.map((r) => this.rowToHit(r, Number(r.score))); // BM25 relevance, higher=better
    return this.scope(hits, opts).slice(0, k);
  }

  /** Hybrid recall: BM25 + vector fused in a SINGLE PASS by infino's native
   *  `hybrid_search` SQL table function — one engine, one query, no rerank
   *  service and no client-side fusion. The TVF runs both retrievers and fuses
   *  them internally, returning rows best-first with a fused `score`. This
   *  first-class single-pass hybrid is the differentiator. Degrades to
   *  vector-only if the query has no usable terms / the FTS index is empty. */
  recallHybrid(query: string, vector: number[], opts: RecallOptions = {}): MemoryHit[] {
    const k = opts.k ?? DEFAULT_K;
    const n = this.fetchN(opts, k);
    const t = this.readTable();
    if (!t) return [];
    const cols = "id, text, importance, category, created_at, score";
    const sql =
      `SELECT ${cols} FROM hybrid_search(` +
      `'${sqlStr(this.table)}', 'text', '${sqlStr(query)}', 'vector', '${vector.join(",")}', ${n})`;
    let rows: Row[];
    try {
      rows = this.db.querySql(sql) as Row[];
    } catch {
      // no usable query terms / empty FTS index -> degrade to vector-only
      return this.recallSemantic(vector, opts);
    }
    const hits = rows.map((r) => this.rowToHit(r, Number(r.score))); // fused score, higher=better
    return this.scope(hits, opts).slice(0, k);
  }

  // ---------- SQL over memory (the view a vector store can't give) ----------
  /** Structured/temporal recall — by time window / category, no similarity. */
  timeline(
    opts: { since?: number; until?: number; category?: MemoryCategory; limit?: number } = {},
  ): MemoryEntry[] {
    const w: string[] = [];
    if (opts.since != null) w.push(`created_at >= ${Number(opts.since)}`);
    if (opts.until != null) w.push(`created_at <= ${Number(opts.until)}`);
    if (opts.category) w.push(`category = '${sqlStr(opts.category)}'`);
    const sql =
      `SELECT id, text, importance, category, created_at FROM ${this.table}` +
      (w.length ? ` WHERE ${w.join(" AND ")}` : "") +
      ` ORDER BY created_at DESC LIMIT ${Number(opts.limit ?? 50)}`;
    return this.safeSql(sql).map((r) => this.toEntry(r));
  }

  /** Escape hatch: arbitrary read-only SQL over the memory table. */
  ask(sql: string): Row[] {
    return this.db.querySql(sql) as Row[];
  }

  // ---------- edit / forget ----------
  forgetById(id: string): boolean {
    return this.forget(`id = '${sqlStr(id)}'`) > 0;
  }
  forget(predicate: string): number {
    const t = this.readTable();
    if (!t) return 0;
    const stats = t.delete(predicate) as { nTombstoned?: number };
    return stats?.nTombstoned ?? 0; // MutationStats: matched / nTombstoned / nNotFound
  }

  count(): number {
    const rows = this.safeSql(`SELECT COUNT(*) AS n FROM ${this.table}`);
    return Number((rows[0] as { n?: unknown })?.n ?? 0);
  }

  // ---------- helpers ----------
  private fetchN(opts: RecallOptions, k: number): number {
    const hasFilter = !!(opts.category || opts.since || opts.until);
    return opts.candidates ?? (hasFilter ? Math.max(20, k * 4) : k);
  }

  /** INTERIM client-side scope filter. Replace with the engine's filtered
   *  vector search (PR #173) once the Node binding exposes VectorFilter — then
   *  the predicate is applied *inside* the kernel (true filtered kNN, no
   *  over-fetch), e.g.:
   *    vectorSearch("vector", vec, k, { projection, filter: { column, mode, query } })
   */
  private scope(hits: MemoryHit[], o: RecallOptions): MemoryHit[] {
    return hits.filter(
      (h) =>
        (o.category ? h.entry.category === o.category : true) &&
        (o.since ? h.entry.createdAt >= o.since : true) &&
        (o.until ? h.entry.createdAt <= o.until : true),
    );
  }
  private rowToHit(r: Row, score: number): MemoryHit {
    return { score, entry: this.toEntry(r) };
  }
  private toEntry(r: Row): MemoryEntry {
    return {
      id: String(r.id),
      text: String(r.text),
      vector: [],
      importance: Number(r.importance ?? 0),
      category: (r.category as MemoryCategory) ?? "other",
      createdAt: Number(r.created_at ?? 0),
    };
  }
  private safeSql(sql: string): Row[] {
    try {
      return this.db.querySql(sql) as Row[];
    } catch {
      return []; // table may not exist before first ingest
    }
  }
}
