import { useCallback, useEffect, useRef, useState } from "react";
import { api, errorText } from "./api";

/** Fetches ``path`` now and every ``ms`` milliseconds (never when ``ms`` is 0). */
export function useData<T>(path: string | null, ms = 0): { data: T | null; error: string | null; reload: () => void } {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tick, setTick] = useState(0);
  const reload = useCallback(() => setTick((t) => t + 1), []);

  useEffect(() => {
    if (!path) return;
    let alive = true;
    const load = () =>
      api
        .get<T>(path)
        .then((value) => {
          if (alive) {
            setData(value);
            setError(null);
          }
        })
        .catch((e) => alive && setError(errorText(e)));
    load();
    const timer = ms ? window.setInterval(load, ms) : undefined;
    return () => {
      alive = false;
      if (timer) window.clearInterval(timer);
    };
  }, [path, ms, tick]);

  return { data, error, reload };
}

/** Runs an action, remembering whether it is busy and its last error. */
export function useAction() {
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const mounted = useRef(true);
  useEffect(
    () => () => {
      mounted.current = false;
    },
    [],
  );

  const run = useCallback(async <T,>(name: string, action: () => Promise<T>): Promise<T | undefined> => {
    setBusy(name);
    setError(null);
    try {
      return await action();
    } catch (e) {
      if (mounted.current) setError(errorText(e));
      return undefined;
    } finally {
      if (mounted.current) setBusy(null);
    }
  }, []);

  return { busy, error, setError, run };
}
