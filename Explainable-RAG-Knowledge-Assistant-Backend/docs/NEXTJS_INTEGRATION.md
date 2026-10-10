# Next.js integration guide — Knowledge Assistant retrieval (v1)

For the team that owns the shared Next.js repository. Integrate these changes through a reviewed PR. Source of truth for the contract: `docs/openapi.json`.

## 1. Rules

1. **The browser never calls the Python service directly.** Browser → Next.js Route Handler (or the team gateway) → FastAPI. The service key stays on the server.
2. **Authenticate the user in Next.js, then authorise the video.** `scope.video_id` is only a retrieval filter. The service does not check ownership: every response returns `scope.ownership_verified: false`. Before forwarding, the Route Handler must confirm that the signed-in user may access that video, using the platform's own data. Never trust a creator or channel ID sent by the client.
3. Don't send unknown fields. The API rejects them with a 422.

## 2. Feature folder

```
src/features/knowledge-assistant/
  api/types.ts              request/response types (below, or generated)
  api/client.ts             fetch wrapper used by UI → calls /api/knowledge-assistant/retrieve
  components/RetrievalResults.tsx
  hooks/useRetrieve.ts
src/app/api/knowledge-assistant/retrieve/route.ts   server-side proxy
```

Other features use the same layout: `src/features/market-research`, `video-prediction`, `analytics`.

## 3. Environment variables (server-only, no `NEXT_PUBLIC_` prefix)

```
KNOWLEDGE_ASSISTANT_API_URL=http://127.0.0.1:8000
KNOWLEDGE_ASSISTANT_API_KEY=<same value as RAG_SERVICE_API_KEY>
KNOWLEDGE_ASSISTANT_TIMEOUT_MS=15000
```

## 4. Types

You can generate these with `npx openapi-typescript docs/openapi.json -o src/features/knowledge-assistant/api/schema.d.ts`, or write them by hand:

```ts
export type RetrievalMethod =
  | "bm25" | "char_tfidf" | "labse"
  | "normalized_bm25" | "normalized_labse"
  | "multiview_bm25" | "multiview_labse" | "multiview_hybrid";

export interface RetrieveRequest {
  query: string;                       // 1..1000 chars, not blank
  method?: RetrievalMethod;            // default "bm25"
  top_k?: number;                      // 1..50, default 10
  scope?: { video_id: string | null }; // null = global research corpus
}

export interface RetrievedComment {
  rank: number;
  document_id: string;
  video_id: string;
  text: string;
  likes: number | null;
  video_url: string | null;
  score: number;                        // compare only within one response
  score_type: "bm25" | "char_tfidf_cosine" | "labse_cosine" | "rrf";
  views: { view: string; rank: number; score: number }[] | null;  // multiview only
}

export interface RetrieveResponse {
  request_id: string;
  query: string;
  method: RetrievalMethod;
  top_k: number;
  scope: { type: "video" | "global"; video_id: string | null; documents_searched: number; ownership_verified: false };
  normalization: { status: string; normalized_query: string; coverage: number; used: boolean } | null;
  results: RetrievedComment[];
  result_count: number;
  notice: string;                       // show it: results are candidates, not verified evidence
}

export interface ApiError {
  error: { code: string; message: string; request_id: string; details?: unknown[] };
}
```

## 5. Route Handler — `src/app/api/knowledge-assistant/retrieve/route.ts`

```ts
import { NextRequest, NextResponse } from "next/server";
import type { RetrieveRequest } from "@/features/knowledge-assistant/api/types";
// Replace with the team's real auth/ownership helpers:
import { getSessionUser, userCanAccessVideo } from "@/lib/auth";

export const runtime = "nodejs";

export async function POST(req: NextRequest) {
  const user = await getSessionUser(req);
  if (!user) return NextResponse.json({ error: { code: "unauthenticated", message: "Sign in required." } }, { status: 401 });

  let input: RetrieveRequest;
  try { input = await req.json(); } catch {
    return NextResponse.json({ error: { code: "invalid_json", message: "Body must be JSON." } }, { status: 400 });
  }

  const videoId = input.scope?.video_id ?? null;
  // Product UI: require a video and check ownership on the server.
  if (!videoId || !(await userCanAccessVideo(user.id, videoId))) {
    return NextResponse.json({ error: { code: "forbidden", message: "No access to this video." } }, { status: 403 });
  }

  const body: RetrieveRequest = {           // forward only the allowed fields
    query: input.query, method: input.method ?? "bm25", top_k: input.top_k ?? 10, scope: { video_id: videoId },
  };

  try {
    const upstream = await fetch(`${process.env.KNOWLEDGE_ASSISTANT_API_URL}/v1/retrieve`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Service-Key": process.env.KNOWLEDGE_ASSISTANT_API_KEY!,
        "X-Request-ID": crypto.randomUUID(),
      },
      body: JSON.stringify(body),
      cache: "no-store",
      signal: AbortSignal.timeout(Number(process.env.KNOWLEDGE_ASSISTANT_TIMEOUT_MS ?? 15000)),
    });
    const data = await upstream.json();
    // A 401 from upstream means the service key is misconfigured. Don't expose it as a user auth error.
    if (upstream.status === 401) return NextResponse.json({ error: { code: "upstream_misconfigured", message: "Service unavailable." } }, { status: 502 });
    return NextResponse.json(data, { status: upstream.status });
  } catch (e) {
    const timeout = e instanceof DOMException && e.name === "TimeoutError";
    return NextResponse.json(
      { error: { code: timeout ? "upstream_timeout" : "upstream_unreachable", message: "Retrieval service unavailable." } },
      { status: timeout ? 504 : 502 });
  }
}
```

## 6. Error handling

| HTTP | `error.code` | UI action |
|---|---|---|
| 200 with `results: []` | — | "No matching comments retrieved". This does not mean no relevant comments exist. Suggest another method. |
| 401 | `unauthorized` | Server key misconfigured. The proxy maps it to 502. |
| 403 | `global_scope_disabled` | Send a `video_id` |
| 404 | `video_not_found` | Video not indexed yet |
| 422 | `invalid_request` | Show field errors from `details[].loc/msg` |
| 503 | `method_unavailable` / `engine_not_ready` | Fall back to `bm25` or retry later |
| 500 | `internal_error` | Generic message; log `request_id` |

Always log `request_id` (also sent back in the `X-Request-ID` header) so frontend and backend logs can be correlated.

## 7. Health

`GET /v1/health` needs no authentication and returns `{status, version, api_version, corpus_documents, videos, semantic: {index_present, loaded}, auth_enabled}`. Use it for gateway readiness checks.
