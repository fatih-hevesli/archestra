import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mockUseQuery = vi.fn();

vi.mock("@archestra/shared", () => ({
  archestraApiSdk: {
    getRetrievalEvaluationCapabilities: vi.fn(),
    listRetrievalEvaluationRuns: vi.fn(),
    getRetrievalEvaluationRun: vi.fn(),
    startRetrievalEvaluation: vi.fn(),
    cancelRetrievalEvaluation: vi.fn(),
    compareRetrievalEvaluations: vi.fn(),
  },
}));

vi.mock("@tanstack/react-query", async () => {
  const actual = await vi.importActual<typeof import("@tanstack/react-query")>(
    "@tanstack/react-query",
  );
  return {
    ...actual,
    useQuery: (...args: unknown[]) => mockUseQuery(...args),
  };
});

import { useRetrievalEvaluationRuns } from "./retrieval-evaluation.query";

describe("useRetrievalEvaluationRuns", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUseQuery.mockImplementation((options) => options);
  });

  it("polls while any durable evaluation run is active", () => {
    const { result } = renderHook(() => useRetrievalEvaluationRuns(), {
      wrapper: createWrapper(),
    });
    expect(result.current).toBeDefined();
    const options = mockUseQuery.mock.calls[0][0] as {
      refetchInterval: (query: {
        state: { data: Array<{ status: string }> };
      }) => number | false;
    };
    expect(
      options.refetchInterval({ state: { data: [{ status: "running" }] } }),
    ).toBe(2_000);
    expect(
      options.refetchInterval({
        state: { data: [{ status: "cancel_requested" }] },
      }),
    ).toBe(2_000);
  });

  it("stops polling when every run is terminal", () => {
    const { result } = renderHook(() => useRetrievalEvaluationRuns(), {
      wrapper: createWrapper(),
    });
    expect(result.current).toBeDefined();
    const options = mockUseQuery.mock.calls[0][0] as {
      refetchInterval: (query: {
        state: { data: Array<{ status: string }> };
      }) => number | false;
    };
    expect(
      options.refetchInterval({
        state: { data: [{ status: "completed" }, { status: "blocked" }] },
      }),
    ).toBe(false);
  });
});

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
}
