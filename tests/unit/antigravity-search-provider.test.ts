import test from "node:test";
import assert from "node:assert/strict";

const {
  SEARCH_CREDENTIAL_FALLBACKS,
  getSearchProvider,
  selectProvider,
  supportsSearchType,
  getSearchCredentialFallbacks,
} = await import("../../open-sse/config/searchRegistry.ts");

const {
  ANTIGRAVITY_SEARCH_PROVIDER_ID,
  DEFAULT_ANTIGRAVITY_SEARCH_MODEL,
  buildAntigravitySearchRequest,
  extractAntigravityGroundingHits,
} = await import("../../open-sse/handlers/search/antigravitySearch.ts");

test("antigravity-search is registered with fallbackOnly: true for web search", () => {
  const cfg = getSearchProvider(ANTIGRAVITY_SEARCH_PROVIDER_ID);
  assert.ok(cfg, "antigravity-search should exist in SEARCH_PROVIDERS");
  assert.equal(cfg!.id, "antigravity-search");
  assert.equal(cfg!.fallbackOnly, true);
  assert.deepEqual(cfg!.searchTypes, ["web"]);
  assert.equal(supportsSearchType(cfg, "web"), true);
  assert.equal(supportsSearchType(cfg, "x"), false);
});

test("auto-select for web NEVER picks antigravity-search (quota protection)", () => {
  const picked = selectProvider(undefined, "web");
  assert.ok(picked);
  assert.notEqual(picked!.id, "antigravity-search");
});

test("auto-select without searchType defaults to web and NEVER picks antigravity-search", () => {
  const picked = selectProvider();
  assert.ok(picked);
  assert.notEqual(picked!.id, "antigravity-search");
});

test("explicit antigravity-search for web is selected", () => {
  const picked = selectProvider("antigravity-search", "web");
  assert.ok(picked);
  assert.equal(picked!.id, "antigravity-search");
});

test("explicit antigravity-search for search_type x is rejected", () => {
  assert.equal(selectProvider("antigravity-search", "x"), null);
});

test("antigravity-search reuses antigravity and agy OAuth credentials", () => {
  assert.deepEqual(getSearchCredentialFallbacks("antigravity-search"), ["antigravity", "agy"]);
  assert.equal("antigravity-search" in SEARCH_CREDENTIAL_FALLBACKS, true);
});

test("buildAntigravitySearchRequest constructs valid Cloud Code grounding payload", () => {
  const cfg = getSearchProvider("antigravity-search")!;
  const req = buildAntigravitySearchRequest(cfg, {
    query: "latest news on fusion energy",
    maxResults: 5,
    token: "ya29.test-oauth-token",
    providerSpecificData: {
      projectId: "test-gcp-project",
    },
  });

  assert.ok(req.url.includes("streamGenerateContent?alt=sse"));
  assert.equal(req.init.method, "POST");
  const headers = req.init.headers as Record<string, string>;
  assert.equal(headers["Authorization"], "Bearer ya29.test-oauth-token");
  assert.equal(headers["Content-Type"], "application/json");

  const body = JSON.parse(req.init.body as string);
  assert.equal(body.project, "test-gcp-project");
  assert.equal(body.model, DEFAULT_ANTIGRAVITY_SEARCH_MODEL);
  assert.ok(Array.isArray(body.request?.tools));
  assert.deepEqual(body.request.tools[0], { googleSearch: {} });
  assert.equal(body.request.contents[0].parts[0].text, "latest news on fusion energy");
});

test("extractAntigravityGroundingHits extracts web chunks and text snippets", () => {
  const sseChunk = {
    response: {
      candidates: [
        {
          content: {
            parts: [{ text: "Commercial fusion energy has reached net energy gain in 2026." }],
          },
          groundingMetadata: {
            groundingChunks: [
              {
                web: {
                  uri: "https://nature.com/articles/fusion-2026",
                  title: "Net Energy Gain in Fusion Experiment",
                },
              },
              {
                web: {
                  uri: "https://science.org/fusion-breakthrough",
                  title: "Milestone in Clean Fusion Power",
                },
              },
            ],
          },
        },
      ],
    },
  };

  const hits = extractAntigravityGroundingHits(sseChunk, "fusion energy", 5);
  assert.equal(hits.length, 2);
  assert.equal(hits[0].url, "https://nature.com/articles/fusion-2026");
  assert.equal(hits[0].title, "Net Energy Gain in Fusion Experiment");
  assert.ok(hits[0].snippet.includes("fusion energy has reached net energy gain"));
});

test("extractAntigravityGroundingHits handles empty grounding gracefully", () => {
  const emptyChunk = {
    response: {
      candidates: [
        {
          content: { parts: [{ text: "No results." }] },
        },
      ],
    },
  };
  const hits = extractAntigravityGroundingHits(emptyChunk, "unknown query", 5);
  assert.deepEqual(hits, []);
});

const { refreshAntigravitySearchToken, tryAntigravitySearchProvider } =
  await import("../../open-sse/handlers/search/antigravitySearch.ts");

test("refreshAntigravitySearchToken keeps a fresh access token without calling refresh", async () => {
  let calls = 0;
  const result = await refreshAntigravitySearchToken(
    {
      accessToken: "fresh-token",
      refreshToken: "refresh-token",
      expiresAt: new Date(Date.now() + 30 * 60 * 1000).toISOString(),
    },
    {
      checkAndRefreshToken: async () => {
        calls += 1;
        return {};
      },
    }
  );
  assert.equal(result.ok, true);
  assert.equal(calls, 0, "a valid token must not rotate the refresh token");
});

test("refreshAntigravitySearchToken refreshes a near-expiry token and merges the result", async () => {
  const seen: string[] = [];
  const result = await refreshAntigravitySearchToken(
    {
      provider: "agy",
      accessToken: "old-token",
      refreshToken: "refresh-token",
      expiresAt: new Date(Date.now() + 10_000).toISOString(),
    },
    {
      checkAndRefreshToken: async (provider, credentials) => {
        seen.push(provider);
        assert.equal(credentials.refreshToken, "refresh-token");
        return {
          accessToken: "rotated-token",
          expiresAt: new Date(Date.now() + 3600_000).toISOString(),
        };
      },
    }
  );
  assert.equal(result.ok, true);
  if (result.ok) assert.equal(result.credentials.accessToken, "rotated-token");
  assert.deepEqual(seen, ["agy"]);
});

test("refreshAntigravitySearchToken surfaces a revoked refresh token as a named failure", async () => {
  const result = await refreshAntigravitySearchToken(
    {
      accessToken: "old-token",
      refreshToken: "revoked",
      expiresAt: new Date(Date.now() - 1000).toISOString(),
    },
    { checkAndRefreshToken: async () => ({ error: "unrecoverable_refresh_error" }) }
  );
  assert.equal(result.ok, false);
  if (!result.ok) assert.match(result.error, /revoked or expired/);
});

test("tryAntigravitySearchProvider dispatches the refreshed bearer through the shared fetch chokepoint", async () => {
  const config = getSearchProvider("antigravity-search")!;
  let capturedAuth = "";
  const result = await tryAntigravitySearchProvider(
    {
      config,
      params: { query: "node release", maxResults: 3 },
      credentials: {
        provider: "agy",
        accessToken: "expired",
        refreshToken: "refresh-token",
        expiresAt: new Date(Date.now() - 1000).toISOString(),
        connectionId: "conn-1",
        projectId: "conn-gcp-project",
      },
      connectionId: "conn-1",
      resolveSearchProxy: async () => ({ proxy: null, proxyLevel: "direct" }),
      executeProviderFetch: async (params) => {
        capturedAuth = String((params.init.headers as Record<string, string>).Authorization);
        return { success: true, data: { results: [] } };
      },
      normalizeResponse: () => ({ results: [], totalResults: 0 }),
    },
    {
      checkAndRefreshToken: async () => ({ accessToken: "rotated-token" }),
    }
  );
  assert.equal((result as { success: boolean }).success, true);
  assert.equal(capturedAuth, "Bearer rotated-token");
});

test("buildAntigravitySearchRequest fails closed without any stored project id", () => {
  const cfg = getSearchProvider("antigravity-search")!;
  assert.throws(
    () =>
      buildAntigravitySearchRequest(cfg, {
        query: "node release",
        maxResults: 3,
        token: "ya29.t",
      }),
    /GCP_PROJECT_REQUIRED/
  );
});

test("tryAntigravitySearchProvider prefers the connection-level projectId over fallbacks", async () => {
  const cfg = getSearchProvider("antigravity-search")!;
  let capturedProject = "";
  const result = await tryAntigravitySearchProvider({
    config: cfg,
    params: { query: "node release", maxResults: 3 },
    credentials: {
      provider: "antigravity",
      accessToken: "live-token",
      expiresAt: new Date(Date.now() + 30 * 60 * 1000).toISOString(),
      projectId: "top-level-project",
      providerSpecificData: { projectId: "psd-project" },
    },
    resolveSearchProxy: async () => ({ proxy: null, proxyLevel: "direct" }),
    executeProviderFetch: async (params) => {
      capturedProject = String(JSON.parse(String(params.init.body)).project);
      return { success: true, data: { results: [] } };
    },
    normalizeResponse: () => ({ results: [], totalResults: 0 }),
  });
  assert.equal((result as { success: boolean }).success, true);
  assert.equal(capturedProject, "top-level-project");
});

test("tryAntigravitySearchProvider fails closed when no Antigravity connection exists", async () => {
  const config = getSearchProvider("antigravity-search")!;
  const result = await tryAntigravitySearchProvider({
    config,
    params: { query: "node release", maxResults: 3 },
    credentials: {},
    resolveSearchProxy: async () => ({ proxy: null, proxyLevel: "direct" }),
    executeProviderFetch: async () => ({ success: true }),
    normalizeResponse: () => ({ results: [], totalResults: 0 }),
  });
  assert.equal((result as { success: boolean }).success, false);
  assert.equal((result as { status: number }).status, 401);
});
