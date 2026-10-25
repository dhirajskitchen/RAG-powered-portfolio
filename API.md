# Portfolio RAG Chatbot API: Integration Guide

A small, stateless HTTP API that answers questions about Dhiraj using only the documents in `context_documents/`. Every factual sentence in a reply carries a citation marker such as `[C0007]`, and the response lists the matching source chunks.

---

## 1. Quick reference

| Item | Value |
|---|---|
| Base URL (local) | `http://localhost:8000` |
| Chat endpoint | `POST /api/chat` |
| Health endpoint | `GET /api/health` |
| Interactive docs | `GET /docs` (Swagger UI) |
| Auth | None (protect with CORS + rate limit; see §8) |
| Content type | `application/json` |
| State | None on the server. The client sends `history` each time |

---

## 2. Running the server

```powershell
pip install -r requirements-api.txt
uvicorn rag_api.app:app --host 0.0.0.0 --port 8000
```

Startup takes about 4 seconds. The index is built in memory from `context_documents/`, so **restart the server after editing the documents**.

### Environment variables (`.env` in the project root)

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `GEMINI_API_KEY` | Yes, unless `GROQ_API_KEY` is set | none | Primary LLM (free tier) |
| `GROQ_API_KEY` | No | none | Fallback LLM if Gemini fails or is rate-limited |
| `GEMINI_CHAT_MODEL` | No | `gemini-2.5-flash-lite` | Gemini model name |
| `GROQ_MODEL` | No | `llama-3.1-8b-instant` | Groq model name |
| `ALLOWED_ORIGINS` | **Set this in production** | `*` | Comma-separated list of sites allowed to call the API, e.g. `https://dhiraj.dev,https://www.dhiraj.dev` |
| `RATE_LIMIT_PER_MIN` | No | `15` | Max chat requests per IP per minute |

---

## 3. `POST /api/chat`

### Request body

```json
{
  "message": "What are your technical skills?",
  "history": [
    { "role": "user", "content": "Tell me about yourself" },
    { "role": "assistant", "content": "I am Dhiraj, a B.Tech student... [C0003]" }
  ]
}
```

| Field | Type | Rules |
|---|---|---|
| `message` | string | **Required.** 1 to 500 characters |
| `history` | array | Optional. At most 8 items; the server uses only the last 4. Each item is `{role, content}` where `role` is `"user"` or `"assistant"` and `content` is at most 1500 characters |

`history` is only used for follow-up questions like "what about that?" and for conversational tone. Send the previous turns exactly as they were displayed (including the citation markers is fine).

### Response body (200)

```json
{
  "reply": "I primarily use Python for AI, ML and Data Science projects, JavaScript for web development, and Java for DSA [C0026]. I am comfortable with LangChain, Scikit-learn, TensorFlow, React and Django [C0028].",
  "citations": [
    {
      "id": "C0026",
      "source": "01_identity_and_background.md",
      "section": "Technical Skills > Programming Languages",
      "snippet": "* Python: ..."
    },
    {
      "id": "C0028",
      "source": "01_identity_and_background.md",
      "section": "Technical Skills > AI / ML",
      "snippet": "..."
    }
  ],
  "grounded": true
}
```

| Field | Meaning |
|---|---|
| `reply` | Answer text in the first person. Contains inline markers like `[C0026]` |
| `citations` | Only the chunks the reply actually cites, in order of first appearance. `snippet` is the first 200 characters of the chunk |
| `grounded` | `true` when the reply is a document-backed answer (or the greeting). `false` when the bot had no information and returned the refusal message |

### The three kinds of reply

1. **Grounded answer**: `grounded: true`, `citations` non-empty.
2. **Greeting / thanks** ("hi", "hello", "thanks"): `grounded: true`, `citations: []`, canned text. No LLM call is made.
3. **Refusal**: `grounded: false`, `citations: []`, reply is `"I don't have information about that in my portfolio knowledge base."` It is returned when nothing in the documents matches, when the model says it cannot answer, or when the model's answer has no valid citation.

Use `grounded` and `citations.length` to style replies. For example, show the "Sources" list only when `citations` is non-empty.

### Error responses

| Status | When | Body | What to do in the UI |
|---|---|---|---|
| `422` | Invalid body (empty or over-long message, bad role, too many history items) | FastAPI validation detail | Validate on the client first (message ≤ 500 chars) |
| `429` | More than `RATE_LIMIT_PER_MIN` requests from one IP in 60 seconds | `{"detail": "Too many requests, please slow down."}` | Show a "please wait" message; disable send for a few seconds |
| `503` | All configured LLM providers failed (quota, outage, bad key) | `{"detail": "The assistant is temporarily unavailable, please try again shortly."}` | Show a friendly retry message |

---

## 4. `GET /api/health`

```json
{ "status": "ok", "chunks": 86 }
```

`chunks` is the number of document chunks indexed. Use it for uptime monitors and to confirm your documents loaded. It does not call the LLM, so it is free.

---

## 5. Calling it from your website

### Plain JavaScript

```js
const API_URL = "https://YOUR-API-HOST";   // no trailing slash
let history = [];                          // keep this per visitor, in memory or sessionStorage

async function ask(message) {
  const res = await fetch(`${API_URL}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, history }),
  });

  if (res.status === 429) throw new Error("You're sending messages too fast. Try again in a moment.");
  if (!res.ok) throw new Error("The assistant is unavailable right now.");

  const data = await res.json();
  history.push({ role: "user", content: message });
  history.push({ role: "assistant", content: data.reply });
  history = history.slice(-8);             // server only needs the last few turns
  return data;
}
```

### React hook

```jsx
import { useState } from "react";

const API_URL = import.meta.env.VITE_CHAT_API_URL;

export function useChat() {
  const [messages, setMessages] = useState([]);   // {role, content, citations?, grounded?}
  const [loading, setLoading] = useState(false);

  async function send(text) {
    const history = messages.slice(-8).map(({ role, content }) => ({ role, content }));
    setMessages((m) => [...m, { role: "user", content: text }]);
    setLoading(true);
    try {
      const res = await fetch(`${API_URL}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, history }),
      });
      if (!res.ok) throw new Error(res.status === 429 ? "Slow down a bit." : "Assistant unavailable.");
      const data = await res.json();
      setMessages((m) => [...m, { role: "assistant", ...{ content: data.reply, citations: data.citations, grounded: data.grounded } }]);
    } catch (e) {
      setMessages((m) => [...m, { role: "assistant", content: e.message, grounded: false, citations: [] }]);
    } finally {
      setLoading(false);
    }
  }

  return { messages, send, loading };
}
```

### Rendering citations

The reply contains markers like `[C0026]`. Two common treatments:

**a) Turn markers into clickable footnote chips.** Match each marker against `citations` by `id`:

```js
function renderReply(reply, citations) {
  const byId = Object.fromEntries(citations.map((c) => [c.id, c]));
  return reply.replace(/\[(C\d{4})\]/g, (m, id) =>
    byId[id]
      ? `<button class="cite" data-id="${id}" title="${byId[id].section}">${id}</button>`
      : ""
  );
}
```

> If you inject HTML like this, **escape the reply text first** (or build DOM nodes instead). The reply is model output and must be treated as untrusted. In React, split on the regex and render components rather than using `dangerouslySetInnerHTML`.

**b) Show a "Sources" list under the message** using `citations[].source`, `section` and `snippet`.

Marker ids (`C0001`...) are numbered by document order at startup. They stay stable until you add, remove or reorder content in `context_documents/`.

### Conversation behaviour to be aware of

- Send `history` so that short follow-ups work. A message under 4 words is combined with the previous user question for retrieval ("and the second one?" → searches the earlier question too).
- A new visitor or a "Clear chat" button should simply reset `history` to `[]`.
- Identical questions are cached on the server for 1 hour, so repeats return instantly and don't use LLM quota.

---

## 6. How it works (for debugging)

```
question ──► greeting? ──yes──► canned reply (no LLM)
                │ no
                ▼
        cache hit? ──yes──► cached reply
                │ no
                ▼
   BM25 search over 86 chunks (top 5, in memory, <1 ms)
                │
        no matches? ──yes──► refusal (no LLM)
                │ no
                ▼
   ONE LLM call: Gemini 2.5 Flash-Lite (Groq fallback)
   prompt = rules + the 5 chunks + last turns + question
                │
                ▼
   Validate: NO_ANSWER → refusal; strip citation ids that
   were not retrieved; no valid citation left → refusal
                │
                ▼
          reply + citations
```

Why this design: the documents total about 20 KB, so a keyword index is enough. It avoids loading embedding and reranker models and keeps startup and per-request latency low. Typical latency is about 1 to 1.5 seconds, almost all of it the LLM call.

Chunking: files in `context_documents/` are split on `#` and `##` headings, YAML blocks are removed, and sections over about 1200 characters are split on paragraph boundaries. The `source` and `section` shown in citations come from the file name and these headings.

---

## 7. Updating the knowledge base

1. Edit or add `.md` files in `context_documents/`. Use clear `#` / `##` headings; the heading text is indexed and shown in citations.
2. Restart the server.
3. Check `GET /api/health`; `chunks` should change.

If the bot says "I don't have information" for something that *is* in the documents, the usual cause is wording. Search is keyword-based, so add the visitor's likely phrasing to the document (the existing "Question Variants" sections work well for this).

---

## 8. Going live checklist

- [ ] **Host it over HTTPS.** A site served over HTTPS cannot call a plain `http://` API (the browser blocks it as mixed content). Free-tier options include Render, Fly.io, Railway or a Hugging Face Space. Note that free instances often sleep when idle, so the first request after a pause can take 30 to 60 seconds.
- [ ] **Set `ALLOWED_ORIGINS`** to your exact site origin(s), including scheme, with no trailing slash and no path. `*` lets any site use your API and your LLM quota.
- [ ] **Set `GEMINI_API_KEY` as a secret in the host's dashboard.** Do not commit `.env` (it is already git-ignored). The key lives only on the server and must never appear in website code.
- [ ] **Add a Groq key** (`GROQ_API_KEY`) as a fallback for when Gemini's free quota is hit.
- [ ] **Run a single worker/instance.** The cache and rate limiter live in process memory. Multiple instances each keep their own, so limits become per-instance.
- [ ] **Behind a proxy or load balancer**, the rate limiter sees the proxy's IP rather than the visitor's, so all visitors share one limit. Raise `RATE_LIMIT_PER_MIN`, or ask me to add trusted `X-Forwarded-For` handling.
- [ ] **Test from the real site**: open the browser dev tools, check the Network tab for CORS errors, then try a question the documents don't cover and confirm the refusal shows correctly.

### Limits to plan for

- Free LLM tiers have daily and per-minute caps. When exhausted, the API returns `503`; make sure your UI handles it.
- The rate limit is per IP only and the API has no authentication. For a public portfolio this is usually enough; for stricter control, add an API key or a proxy that holds the key.

---

## 9. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Browser console: "blocked by CORS policy" | The site's origin is not in `ALLOWED_ORIGINS`, or the value has a trailing slash or the wrong scheme (`http` vs `https`, `www` vs bare domain) |
| "Mixed content" error | Website is HTTPS but `API_URL` is `http://`. Deploy the API behind HTTPS |
| `{"detail":"Not Found"}` | You opened `/` in a browser. Use `/docs` or `/api/health`, and make sure the client calls `/api/chat` |
| `503` on every request | Missing or invalid `GEMINI_API_KEY`, or quota exhausted. Check the server log line starting `LLM failure:` |
| `429` during testing | You exceeded `RATE_LIMIT_PER_MIN`. Wait a minute or raise it |
| `422` | Message empty or over 500 characters, or `history` roles other than `user` / `assistant` |
| Always "I don't have information…" | The documents lack the topic, or the wording doesn't overlap with the question (see §7) |
| Reply has no citation chips | Check `citations` is non-empty. The refusal and greeting replies carry none by design |
| Document edits not showing | The server was not restarted |
| First request very slow in production | Free host was asleep (cold start) |

---

## 10. File map

| Path | Role |
|---|---|
| `rag_api/app.py` | FastAPI app, CORS, rate limiting, cache, prompt, citation validation |
| `rag_api/retriever.py` | Markdown chunking + BM25 index and search |
| `rag_api/llm.py` | Gemini call with Groq fallback |
| `context_documents/` | The only source of truth for answers |
| `requirements-api.txt` | Minimal runtime dependencies for deployment |
| `main.py` | Older, separate prototype with a hardcoded persona. **Not** part of this API |
