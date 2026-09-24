/**
 * Antigravity Google Search Grounding for POST /v1/search (Issue #14654).
 *
 * Routes a search query through Gemini `googleSearch` grounding using an existing
 * Antigravity OAuth connection. Kept in this dedicated submodule to keep
 * open-sse/handlers/search.ts strictly under the frozen file-size ratchet.
 */

import type { SearchProviderConfig } from "../../config/searchRegistry.ts";
import { ANTIGRAVITY_RUNTIME_BASE_URLS } from "../../config/antigravityUpstream.ts";
import {
  generateAntigravityRequestId,
  generateAntigravitySessionId,
  getAntigravityEnvelopeUserAgent,
} from "../../services/antigravityIdentity.ts";
import { getAntigravityContentHeaders } from "../../services/antigravityHeaders.ts";
import {
  applyAntigravityClientProfileHeaders,
  getAntigravityClientProfile,
} from "../../services/antigravityClientProfile.ts";
import { scrubProxyAndFingerprintHeaders } from "../../services/antigravityHeaderScrub.ts";
import type {
  ExecuteProviderFetchParams,
  ProviderFetchResult,
  ResolvedSearchProxy,
} from "./searchProxy.ts";
import type { SearchResult } from "../search.ts";

export const ANTIGRAVITY_SEARCH_PROVIDER_ID = "antigravity-search";
export const DEFAULT_ANTIGRAVITY_SEARCH_MODEL = "gemini-3.8-flash-high";
export const ANTIGRAVITY_SEARCH_STREAM_PATH = "/v1internal:streamGenerateContent?alt=sse";

export interface AntigravitySearchParams {
  query: string;
  maxResults: number;
  token?: string;
  projectId?: string;
  providerOptions?: Record<string, unknown>;
  providerSpecificData?: Record<string, unknown>;
}

export type AntigravityGroundingHit = {
  title: string;
  url: string;
  snippet: string;
  source_type?: string;
};

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

/**
 * Extracts grounding citations and web search chunks from a Gemini / Cloud Code response.
 * Handles:
 *   1) Direct JSON with candidate.groundingMetadata
 *   2) SSE stream payload string containing data: {...} lines
 *   3) Array of parsed chunk objects
 */
export function extractAntigravityGroundingHits(
  data: unknown,
  query: string,
  maxResults: number
): AntigravityGroundingHit[] {
  let chunksToInspect: unknown[] = [];

  if (typeof data === "string") {
    // Collect JSON objects from SSE data lines
    const lines = data.split("\n");
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed.startsWith("data:") || trimmed === "data: [DONE]") continue;
      try {
        chunksToInspect.push(JSON.parse(trimmed.slice(5).trim()));
      } catch {
        // Skip unparseable SSE line
      }
    }
  } else if (Array.isArray(data)) {
    chunksToInspect = data;
  } else if (data && typeof data === "object") {
    chunksToInspect = [data];
  }

  const hits: AntigravityGroundingHit[] = [];
  const seenUrls = new Set<string>();
  let fullText = "";

  for (const item of chunksToInspect) {
    const root = asRecord(item) ?? {};
    const response = asRecord(root.response) ?? root;
    const candidates = Array.isArray(response.candidates) ? response.candidates : [];

    for (const cand of candidates) {
      const cRec = asRecord(cand);
      if (!cRec) continue;

      const content = asRecord(cRec.content);
      if (Array.isArray(content?.parts)) {
        for (const part of content.parts) {
          const pRec = asRecord(part);
          if (typeof pRec?.text === "string" && !pRec?.thought) {
            fullText += (fullText ? " " : "") + pRec.text.trim();
          }
        }
      }

      const grounding = asRecord(cRec.groundingMetadata || cRec.grounding_metadata);
      if (!grounding) continue;

      const rawChunks = grounding.groundingChunks || grounding.grounding_chunks;
      const chunks: unknown[] = Array.isArray(rawChunks) ? rawChunks : [];

      for (const chunk of chunks) {
        const chRec = asRecord(chunk);
        const web = asRecord(chRec?.web);
        if (!web) continue;
        const url = typeof web.uri === "string" ? web.uri.trim() : "";
        if (!url || !url.startsWith("http") || seenUrls.has(url)) continue;
        seenUrls.add(url);

        const title = typeof web.title === "string" && web.title.trim() ? web.title.trim() : url;
        hits.push({
          title,
          url,
          snippet: fullText ? fullText.slice(0, 500) : query,
          source_type: "google-grounding",
        });
        if (hits.length >= maxResults) break;
      }
      if (hits.length >= maxResults) break;
    }
    if (hits.length >= maxResults) break;
  }

  return hits;
}

/**
 * Normalizes Antigravity search response into standard SearchResult array.
 */
export function normalizeAntigravitySearchResponse(
  data: unknown,
  query: string,
  _searchType: string,
  makeResult: (
    providerId: string,
    item: { title?: string; url?: string; snippet?: string; source_type?: string },
    idx: number,
    now: string
  ) => SearchResult
): { results: SearchResult[]; totalResults: number | null } {
  const now = new Date().toISOString();
  const hits = extractAntigravityGroundingHits(data, query, 20);
  const results = hits.map((hit, idx) =>
    makeResult(
      ANTIGRAVITY_SEARCH_PROVIDER_ID,
      {
        title: hit.title,
        url: hit.url,
        snippet: hit.snippet,
        source_type: hit.source_type,
      },
      idx,
      now
    )
  );
  return { results, totalResults: results.length };
}

/**
 * Build request envelope for the Antigravity v1internal:streamGenerateContent endpoint.
 */
export function buildAntigravitySearchRequest(
  config: SearchProviderConfig,
  params: AntigravitySearchParams
): { url: string; init: RequestInit } {
  const model =
    (typeof params.providerSpecificData?.model === "string" &&
      params.providerSpecificData.model.trim()) ||
    (typeof params.providerOptions?.model === "string" && params.providerOptions.model.trim()) ||
    DEFAULT_ANTIGRAVITY_SEARCH_MODEL;

  const rawProjectId =
    (typeof params.projectId === "string" && params.projectId.trim()) ||
    (typeof params.providerSpecificData?.projectId === "string" &&
      params.providerSpecificData.projectId.trim()) ||
    (typeof params.providerOptions?.projectId === "string" &&
      params.providerOptions.projectId.trim()) ||
    "";

  if (!rawProjectId) {
    throw new Error(
      "GCP_PROJECT_REQUIRED: Missing Google Cloud project for Antigravity connection. " +
        "Ensure the Antigravity OAuth connection has completed setup or configure a Project ID."
    );
  }
  const projectId = rawProjectId;

  const clientProfile = getAntigravityClientProfile({
    providerSpecificData: params.providerSpecificData,
  });

  const envelope = {
    project: projectId,
    requestId: generateAntigravityRequestId(),
    model,
    userAgent: getAntigravityEnvelopeUserAgent(),
    requestType: "agent",
    request: {
      contents: [
        {
          role: "user",
          parts: [{ text: params.query }],
        },
      ],
      tools: [{ googleSearch: {} }],
      sessionId: generateAntigravitySessionId(),
      generationConfig: {
        maxOutputTokens: 1024,
        temperature: 0.1,
      },
    },
  };

  // Mirror the native executor's header pipeline (executeAttempt.ts): content
  // headers per client profile, scrub proxy/fingerprint leaks, then apply the
  // profile headers (x-goog-user-project from body.project) exactly like a chat
  // request so Cloud Code sees a first-party envelope.
  const baseHeaders = {
    ...getAntigravityContentHeaders(clientProfile, params.token),
    Accept: "text/event-stream, application/json",
  };
  const scrubbedHeaders = scrubProxyAndFingerprintHeaders(baseHeaders);
  applyAntigravityClientProfileHeaders(
    scrubbedHeaders,
    { providerSpecificData: params.providerSpecificData },
    envelope
  );

  const baseUrl = (config.baseUrl || ANTIGRAVITY_RUNTIME_BASE_URLS[0]).replace(/\/+$/, "");
  const targetUrl = `${baseUrl}${ANTIGRAVITY_SEARCH_STREAM_PATH}`;

  return {
    url: targetUrl,
    init: {
      method: "POST",
      headers: scrubbedHeaders,
      body: JSON.stringify(envelope),
    },
  };
}

export interface TryAntigravitySearchParams {
  config: SearchProviderConfig;
  params: AntigravitySearchParams;
  credentials: Record<string, unknown>;
  log?: {
    info: (tag: string, message: string) => void;
    error: (tag: string, message: string) => void;
    warn?: (tag: string, message: string) => void;
  } | null;
  connectionId?: string;
  apiKeyId?: string;
  resolveSearchProxy: (
    connectionId: string | undefined,
    apiKeyId: string | undefined,
    providerId: string
  ) => Promise<ResolvedSearchProxy>;
  executeProviderFetch: (params: ExecuteProviderFetchParams) => Promise<ProviderFetchResult>;
  normalizeResponse: ExecuteProviderFetchParams["normalize"];
}

/** Injectable token-refresh dependency (tests pass a fake; runtime uses the shared refresher). */
export interface AntigravityTokenRefreshDeps {
  checkAndRefreshToken?: (
    provider: string,
    credentials: Record<string, unknown>
  ) => Promise<Record<string, unknown>>;
}

const TOKEN_EXPIRY_SAFETY_MS = 60_000;

async function defaultCheckAndRefreshToken(
  provider: string,
  credentials: Record<string, unknown>
): Promise<Record<string, unknown>> {
  const { checkAndRefreshToken } = await import("@/sse/services/tokenRefresh.ts");
  return checkAndRefreshToken(provider, credentials);
}

/**
 * Refresh the reused antigravity/agy OAuth access token when it is missing or
 * about to expire. Uses the same checkAndRefreshToken path as the chat surface:
 * expiry-lead-gated (no unconditional rotation), per-connection mutex, DB
 * staleness re-check, and atomic persistence of rotated refresh tokens.
 *
 * Returns the credentials to use, or an unrecoverable state that must surface
 * as a named 401 instead of dispatching an expired bearer to Cloud Code.
 */
export async function refreshAntigravitySearchToken(
  credentials: Record<string, unknown>,
  deps: AntigravityTokenRefreshDeps = {}
): Promise<{ ok: true; credentials: Record<string, unknown> } | { ok: false; error: string }> {
  const token =
    typeof credentials.accessToken === "string" && credentials.accessToken
      ? credentials.accessToken
      : null;
  const expiresAtMs =
    typeof credentials.expiresAt === "string" ? Date.parse(credentials.expiresAt) : NaN;
  const nearExpiry =
    Number.isFinite(expiresAtMs) && expiresAtMs - Date.now() < TOKEN_EXPIRY_SAFETY_MS;

  if (token && !nearExpiry) return { ok: true, credentials };
  if (typeof credentials.refreshToken !== "string" || !credentials.refreshToken) {
    return {
      ok: false,
      error: token
        ? "Antigravity OAuth access token is expired and no refresh token is stored. Reconnect the Antigravity provider connection."
        : "No active Antigravity OAuth connection available for antigravity-search.",
    };
  }

  const refreshProvider =
    typeof credentials.provider === "string" && credentials.provider
      ? credentials.provider
      : "antigravity";
  const refresher = deps.checkAndRefreshToken ?? defaultCheckAndRefreshToken;
  let refreshed: Record<string, unknown>;
  try {
    refreshed = await refresher(refreshProvider, credentials);
  } catch (err) {
    return {
      ok: false,
      error: `Antigravity OAuth refresh failed: ${err instanceof Error ? err.message : String(err)}`,
    };
  }

  if (typeof refreshed?.accessToken === "string" && refreshed.accessToken) {
    if (refreshed.accessToken === token && nearExpiry) {
      return {
        ok: false,
        error:
          "Antigravity OAuth refresh token is revoked or expired. Reconnect the Antigravity provider connection.",
      };
    }
    return { ok: true, credentials: { ...credentials, ...refreshed } };
  }
  return {
    ok: false,
    error:
      "Antigravity OAuth refresh token is revoked or expired. Reconnect the Antigravity provider connection.",
  };
}

/**
 * Dedicated dispatch for antigravity-search (#14654).
 *
 * Refreshes the reused antigravity/agy OAuth access token when needed, then
 * delegates to the shared search fetch chokepoint so proxy resolution, call
 * logs, and connection cooldowns behave exactly like every other provider.
 */
function recordSearchFailure(
  config: SearchProviderConfig,
  query: string,
  connectionId: string | undefined,
  status: number,
  error: string,
  startTime: number
): void {
  void import("@/lib/usageDb")
    .then(({ saveCallLog }) =>
      saveCallLog({
        method: config.method,
        path: "/v1/search",
        model: config.id,
        provider: config.id,
        connectionId: connectionId || null,
        requestType: "search",
        requestBody: { query: query.slice(0, 200), search_type: "web" },
        status,
        duration: Date.now() - startTime,
        error: error.slice(0, 500),
      })
    )
    .catch(() => undefined);
}

export async function tryAntigravitySearchProvider(
  args: TryAntigravitySearchParams,
  deps: AntigravityTokenRefreshDeps = {}
): Promise<ProviderFetchResult> {
  const { config, credentials, log, connectionId, apiKeyId } = args;
  const startTime = Date.now();
  const fail = (status: number, error: string): ProviderFetchResult => {
    recordSearchFailure(config, args.params.query, connectionId, status, error, startTime);
    return { success: false, status, error };
  };

  const refreshed = await refreshAntigravitySearchToken(credentials, deps);
  if (!refreshed.ok) {
    const failure = refreshed as { ok: false; error: string };
    return fail(401, failure.error);
  }
  const activeCredentials = refreshed.credentials;
  const providerSpecificData =
    activeCredentials.providerSpecificData &&
    typeof activeCredentials.providerSpecificData === "object"
      ? (activeCredentials.providerSpecificData as Record<string, unknown>)
      : undefined;
  const topLevelProjectId =
    typeof activeCredentials.projectId === "string" && activeCredentials.projectId.trim()
      ? activeCredentials.projectId.trim()
      : undefined;
  const params: AntigravitySearchParams = {
    ...args.params,
    token: activeCredentials.accessToken as string | undefined,
    projectId: topLevelProjectId || args.params.projectId,
    providerSpecificData,
  };

  if (!params.token) {
    return fail(401, "No active Antigravity OAuth connection available for antigravity-search.");
  }

  let built: { url: string; init: RequestInit };
  try {
    built = buildAntigravitySearchRequest(config, params);
  } catch (err) {
    return fail(
      400,
      err instanceof Error ? err.message : "Invalid Antigravity search configuration"
    );
  }

  const { proxy, proxyLevel } = await args.resolveSearchProxy(connectionId, apiKeyId, config.id);
  const timeout = Math.min(config.timeoutMs, 60_000);
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);

  log?.info?.("SEARCH", `${config.id} | query: "${params.query.slice(0, 80)}"`);

  return args.executeProviderFetch({
    config,
    url: built.url,
    init: built.init,
    controller,
    timer,
    query: params.query,
    searchType: "web",
    maxResults: params.maxResults,
    startTime,
    connectionId,
    proxy,
    proxyLevel,
    log: log ?? undefined,
    normalize: args.normalizeResponse,
  });
}
