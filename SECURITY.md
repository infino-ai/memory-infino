# Security Policy

## Reporting a vulnerability

Please report security issues **privately** — do not open a public issue for a
suspected vulnerability.

Use GitHub's private vulnerability reporting: open the repository's **Security**
tab and click **"Report a vulnerability."** We aim to acknowledge reports within
a few business days.

## Data handling

memory-infino persists and retrieves agent memory. What that means for your data:

- **Memory is stored where you point it.** Entries live at the configured
  `dbPath` — a local path or *your own* `s3://` / `gs://` / `az://` bucket
  (`storageOptions`). Nothing is sent to Infino; there is no managed service and
  no telemetry.
- **Embeddings depend on the provider you choose.** The default `local`
  embedder (Hugging Face transformers.js) runs in-process — text never leaves
  the machine. If you select `ollama` or an `openai`-compatible provider, the
  text being embedded is sent to that endpoint you configure.
- **Auto-capture stores salient user statements automatically.** With
  `autoCapture` on (the default), messages matching the capture heuristic are
  embedded and stored without an explicit tool call. Disable it
  (`autoCapture: false`) or scope `dbPath` if you don't want automatic
  persistence of conversation content.
- **Recalled memories are treated as untrusted.** Injected memory context is
  wrapped in a `<relevant-memories>` block instructing the agent not to follow
  instructions found inside it (defense against memory-borne prompt injection).
- **Credentials stay in the environment.** Storage and embedding credentials are
  read from the config/environment you provide and are never logged.

## Supported versions

Security fixes are released against the latest published version on npm
([`@infino-ai/memory-infino`](https://www.npmjs.com/package/@infino-ai/memory-infino)).
