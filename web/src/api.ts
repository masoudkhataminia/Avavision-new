export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly body: Record<string, unknown>,
  ) {
    super(message);
  }
}

async function request<T>(method: string, path: string, body?: unknown, raw?: Blob): Promise<T> {
  const init: RequestInit = { method };
  if (raw !== undefined) {
    init.body = raw;
    init.headers = { "Content-Type": "application/octet-stream" };
  } else if (body !== undefined) {
    init.body = JSON.stringify(body);
    init.headers = { "Content-Type": "application/json" };
  }
  const response = await fetch(path, init);
  const text = await response.text();
  const data = text ? JSON.parse(text) : null;
  if (!response.ok) {
    const message = (data && (data.error || data.detail)) || response.statusText;
    throw new ApiError(typeof message === "string" ? message : JSON.stringify(message), response.status, data ?? {});
  }
  return data as T;
}

export const api = {
  get: <T>(path: string) => request<T>("GET", path),
  post: <T>(path: string, body?: unknown) => request<T>("POST", path, body ?? {}),
  put: <T>(path: string, body: unknown) => request<T>("PUT", path, body),
  delete: <T>(path: string) => request<T>("DELETE", path),
  upload: <T>(path: string, file: Blob, method = "POST") => request<T>(method, path, undefined, file),
};

export function errorText(error: unknown): string {
  if (error instanceof ApiError) {
    const compartments = error.body.compartments as { row: number; column: number }[] | undefined;
    return compartments?.length ? `${error.message} (${compartments.length} compartments)` : error.message;
  }
  return error instanceof Error ? error.message : String(error);
}
