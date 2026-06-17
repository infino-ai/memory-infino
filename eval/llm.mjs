// SPDX-License-Identifier: Apache-2.0
// Minimal OpenAI-compatible chat client for the LOCOMO eval (answer + judge).
// Provider-agnostic via env — works with Azure Foundry (/openai/v1), OpenAI, or
// any OpenAI-compatible endpoint. No keys in the repo.
//
//   OPENAI_BASE_URL   e.g. https://<resource>.openai.azure.com/openai/v1
//   OPENAI_API_KEY    the key (sent as both Bearer and api-key; endpoints ignore the unused one)
//   EVAL_MODEL        model/deployment id (default: gpt-5.4)
const BASE = (process.env.OPENAI_BASE_URL || "https://api.openai.com/v1").replace(/\/+$/, "");
const KEY = process.env.OPENAI_API_KEY || "";
export const MODEL = process.env.EVAL_MODEL || "gpt-5.4";

/** One chat completion. Returns { text, promptTokens, completionTokens }. */
export async function chat(messages) {
  const res = await fetch(`${BASE}/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(KEY ? { Authorization: `Bearer ${KEY}`, "api-key": KEY } : {}),
    },
    body: JSON.stringify({ model: MODEL, messages }),
  });
  if (!res.ok) throw new Error(`LLM ${res.status}: ${(await res.text()).slice(0, 200)}`);
  const json = await res.json();
  const u = json.usage ?? {};
  return {
    text: json.choices?.[0]?.message?.content ?? "",
    promptTokens: u.prompt_tokens ?? 0,
    completionTokens: u.completion_tokens ?? 0,
  };
}
