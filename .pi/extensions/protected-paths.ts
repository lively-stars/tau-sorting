/**
 * protected-paths.ts — guardrail that blocks accidental destruction of this repo's
 * expensive / regenerable / runtime-critical files.
 *
 * What it protects (the big gitignored inputs + the webapp runtime files):
 *   - ODF_format.npy, ODF_nc_format.nc   (~2.8GB ODF; slow to re-convert)
 *   - continuumabs.dat, continuumscat.dat, continuumall.dat  (source continuum data)
 *   - data/                              (~2.8GB of reference tables, whole tree)
 *   - .webapp.pid, webapp.log            (make start/stop/status depend on these)
 *
 *   `models/*.dat` is deliberately NOT protected — it's tracked, small source.
 *
 * Behavior:
 *   - `write` / `edit` tool on any protected path  -> blocked.
 *   - `bash` tool that is destructive (rm/rmdir/unlink/shred/mv, tee, sed -i, or a
 *     truncate `>` redirect) AND mentions a protected path -> blocked.
 *   - Reading (read tool; cat/tail/ls; cp FROM a protected file) is always allowed.
 *   - The gate FAILS OPEN: any unexpected error returns "allow" so it can never wedge
 *     the agent. It is a guardrail, not a security boundary.
 *
 * There is no in-agent bypass (by design). If you genuinely need to delete/overwrite
 * one of these, run it yourself outside the agent, or drop this file from
 * .pi/settings.json and restart pi.
 *
 * Load: enabled via .pi/settings.json -> { "extensions": ["extensions/protected-paths.ts"] }
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

// Protected file basenames (matched anywhere a path token resolves to this basename).
const PROTECTED_FILES = new Set([
  "ODF_format.npy",
  "ODF_nc_format.nc",
  "continuumabs.dat",
  "continuumscat.dat",
  "continuumall.dat",
  ".webapp.pid",
  "webapp.log",
]);

// Short "why" for each, surfaced in the block reason so the model understands.
const NOTES: Record<string, string> = {
  "ODF_format.npy": "~2.8GB cached ODF; regenerate via `convert-odf` (slow)",
  "ODF_nc_format.nc": "source ODF NetCDF (~2.8GB)",
  "continuumabs.dat": "source continuum absorption data",
  "continuumscat.dat": "source continuum scattering data",
  "continuumall.dat": "source continuum combined data",
  ".webapp.pid": "Q_rad webapp PID file — make start/stop/status depend on it",
  "webapp.log": "Q_rad webapp log — make start writes here",
};

interface Hit {
  label: string;
  note: string;
}

// strip shell metacharacters and a leading ./ so tokens like `>ODF_format.npy`,
// `data/x,` or `./.webapp.pid` still match.
function cleanToken(raw: string): string {
  return raw.replace(/^\.\/+/, "").replace(/["';,()|&<>]/g, "").trim();
}

// Return the protected path a string references (by basename or the data/ tree), or null.
function protectedHit(text: string): Hit | null {
  for (const raw of text.split(/\s+/)) {
    const t = cleanToken(raw);
    if (!t) continue;
    const base = t.split("/").pop() ?? t;
    if (PROTECTED_FILES.has(base)) {
      return { label: base, note: NOTES[base] ?? "protected file" };
    }
    if (
      t === "data" ||
      t.endsWith("/data") || // absolute/relative path to the dir itself (no trailing slash)
      t.startsWith("data/") ||
      t.startsWith("./data/") ||
      t.includes("/data/")
    ) {
      return { label: "data/", note: "~2.8GB of gitignored reference tables" };
    }
  }
  return null;
}

// Does a bash command perform a destructive/overwriting operation?
function isDestructive(cmd: string): boolean {
  if (/\b(rm|rmdir|unlink|shred|mv)\b/.test(cmd)) return true;
  if (/\btee\b/.test(cmd)) return true; // tee overwrites by default
  if (/\bsed\b/.test(cmd) && /(^|\s)-i(\s|$)|--in-place/.test(cmd)) return true; // sed -i
  if (/(^|[^>])>[^>]/.test(cmd)) return true; // a truncate `>` redirect (not `>>`)
  return false;
}

function block(ctx: { hasUI: boolean; ui: { notify: (m: string, l: "warning") => void } }, reason: string) {
  if (ctx.hasUI) ctx.ui.notify(`Blocked: ${reason}`, "warning");
  return {
    block: true,
    reason:
      `Blocked by protected-paths gate: ${reason}. ` +
      `If you genuinely need this, ask the user to run it directly or disable the gate.`,
  };
}

export default function (pi: ExtensionAPI) {
  pi.on("tool_call", async (event, ctx) => {
    try {
      if (event.toolName === "write" || event.toolName === "edit") {
        const path = String((event.input as { path?: unknown }).path ?? "");
        const hit = protectedHit(path);
        if (hit) {
          return block(ctx, `${event.toolName} of protected path "${hit.label}" (${hit.note})`);
        }
        return undefined;
      }

      if (event.toolName === "bash") {
        const command = String((event.input as { command?: unknown }).command ?? "");
        const hit = protectedHit(command);
        if (hit && isDestructive(command)) {
          return block(ctx, `destructive bash op on protected path "${hit.label}" (${hit.note})`);
        }
        return undefined;
      }

      return undefined;
    } catch {
      // Fail open: a buggy guard must never wedge the agent.
      return undefined;
    }
  });
}
