/**
 * tausort-tools.ts — project tools for the tau-sorting repo.
 *
 * Registers four small tools that wrap the repo's documented commands:
 *   - webapp   : start/stop/restart/status/logs/sync for the Q_rad explorer (Makefile, port 8771)
 *   - test     : run the test_*.py suite via `uv run` (unittest-based via -m unittest,
 *                script-style like test_derivatives.py directly)
 *   - lint     : ruff format + check via scripts/precommit.sh (action: fix | check)
 *   - build-c  : make / make clean / rebuild for the C reference (tausort.x)
 *
 * Also overrides the built-in `bash` tool with a spawn hook that auto-prefixes bare
 * `python` / `python3` / `pytest` / `ruff` / `basedpyright` / `pyright` commands with
 * `uv run`, so the project's uv-managed environment is always used. Commands that
 * already use `uv`/`uvx`, invoke the venv interpreter by path, or call `make` /
 * scripts are left untouched.
 *
 * Loading (pick one):
 *   one-shot:  pi -e .pi/extensions/tausort-tools.ts
 *   persisted: { "extensions": ["extensions/tausort-tools.ts"] } in .pi/settings.json
 *
 * The `@earendil-works/*` and typebox imports are core packages pi bundles for
 * extensions, so this file needs no package.json of its own.
 */

import { StringEnum, Type } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import {
  DEFAULT_MAX_BYTES,
  DEFAULT_MAX_LINES,
  createBashTool,
  formatSize,
  truncateTail,
} from "@earendil-works/pi-coding-agent";
import { Text } from "@earendil-works/pi-tui";
import { existsSync, readFileSync, readdirSync } from "node:fs";
import { basename, resolve } from "node:path";

// --- find the repo root: nearest ancestor (incl. cwd) with both Makefile + pyproject.toml
function findRoot(start: string): string {
  let dir = resolve(start);
  for (let i = 0; i < 16; i++) {
    if (existsSync(resolve(dir, "Makefile")) && existsSync(resolve(dir, "pyproject.toml"))) {
      return dir;
    }
    const parent = resolve(dir, "..");
    if (parent === dir) break;
    dir = parent;
  }
  return resolve(start);
}

// --- is a test file unittest-based? sniff the first few KB for an unittest import.
function isUnitTestFile(root: string, file: string): boolean {
  try {
    const head = readFileSync(resolve(root, file), "utf8").slice(0, 4096);
    return /(^|\n)\s*(import\s+unittest\b|from\s+unittest\b)/.test(head);
  } catch {
    return true; // CLAUDE.md default is `python -m unittest`
  }
}

// --- run a command via pi.exec; return tail-truncated stdout+stderr + exit code
async function run(
  pi: ExtensionAPI,
  command: string,
  args: string[],
  opts: { cwd: string; signal?: AbortSignal; timeout?: number },
) {
  const res = await pi.exec(command, args, {
    cwd: opts.cwd,
    signal: opts.signal,
    timeout: opts.timeout,
  });
  const out = res.stdout ?? "";
  const err = res.stderr ?? "";
  const combined = out + (err ? (out && !out.endsWith("\n") ? "\n" : "") + err : "");
  const trunc = truncateTail(combined, { maxLines: DEFAULT_MAX_LINES, maxBytes: DEFAULT_MAX_BYTES });
  let text = trunc.content.trim();
  if (trunc.truncated) {
    text +=
      `\n\n[output truncated: showing last ${trunc.outputLines}/${trunc.totalLines} lines ` +
      `(${formatSize(trunc.outputBytes)} of ${formatSize(trunc.totalBytes)})]`;
  }
  return { code: res.code, killed: res.killed, text, empty: combined.trim().length === 0 };
}

export default function (pi: ExtensionAPI) {
  // ----------------------------------------------------------------- webapp --
  pi.registerTool({
    name: "webapp",
    label: "Q_rad webapp",
    description:
      "Manage the Q_rad explorer web app (webapp/server.py, port 8771) via the project Makefile. " +
      "Actions: 'start', 'stop', 'restart', 'status', 'logs' (tail webapp.log), 'sync' (uv sync to " +
      "create/refresh ./.venv). The first 'start' after sync reads the ODF (~10-30s). 'start' fails " +
      "with a hint to run 'sync' if ./.venv is missing.",
    promptSnippet: "Start/stop/status the Q_rad web app (make start|stop|restart|status; port 8771)",
    parameters: Type.Object({
      action: StringEnum(["start", "stop", "restart", "status", "logs", "sync"] as const),
    }),
    async execute(_id, params, signal, _onUpdate, ctx) {
      const root = findRoot(ctx.cwd);
      const map: Record<string, [string, string[]]> = {
        start: ["make", ["start"]],
        stop: ["make", ["stop"]],
        restart: ["make", ["restart"]],
        status: ["make", ["status"]],
        sync: ["uv", ["sync"]],
        logs: ["tail", ["-n", "200", "webapp.log"]],
      };
      const [cmd, args] = map[params.action];
      const r = await run(pi, cmd, args, { cwd: root, signal });
      const ok = r.code === 0;
      const body =
        params.action === "logs"
          ? r.empty
            ? "(webapp.log is empty — the server may not have logged yet; try 'status')"
            : r.text
          : `$ ${cmd} ${args.join(" ")}  [exit ${r.code}]` + (r.empty ? "" : "\n" + r.text);
      return {
        content: [{ type: "text", text: body }],
        details: { action: params.action, code: r.code, ok },
      };
    },
    renderCall(args, theme) {
      return new Text(
        theme.fg("toolTitle", theme.bold("webapp ")) + theme.fg("accent", String(args.action)),
        0,
        0,
      );
    },
  });

  // ------------------------------------------------------------------- test --
  pi.registerTool({
    name: "test",
    label: "Run tests",
    description:
      "Run the project's tests with `uv run`. By default runs every test_*.py: unittest-based " +
      "files via `uv run python -m unittest`, and script-style files (e.g. test_derivatives.py) " +
      "directly. Pass `file` to run a single test file. NOTE: some tests need the gitignored ODF / " +
      "continuum / data/ inputs and will error if those are absent.",
    promptSnippet: "Run the test suite (test_*.py) via uv run",
    parameters: Type.Object({
      file: Type.Optional(
        Type.String({
          description: "A single test file to run, e.g. 'test_qrad_optimize.py'. Omit to run all.",
        }),
      ),
    }),
    async execute(_id, params, signal, _onUpdate, ctx) {
      const root = findRoot(ctx.cwd);
      const all = readdirSync(root)
        .filter((n) => /^test_.*\.py$/.test(n))
        .sort();
      const files = params.file
        ? all.includes(basename(params.file))
          ? [basename(params.file)]
          : [params.file]
        : all;
      if (!files.length) {
        return {
          content: [{ type: "text", text: `No test_*.py files found in ${root}` }],
          details: { ran: 0, failures: 0, ok: false },
        };
      }
      const lines: string[] = [];
      let failures = 0;
      for (const f of files) {
        const unit = isUnitTestFile(root, f);
        const args = unit ? ["run", "python", "-m", "unittest", f] : ["run", "python", f];
        const r = await run(pi, "uv", args, { cwd: root, signal });
        const ok = r.code === 0;
        if (!ok) failures++;
        lines.push(`── ${f}  (${unit ? "unittest" : "script"})  exit ${r.code} ${ok ? "✓" : "✗"}`);
        if (r.text) lines.push(r.text);
        lines.push("");
      }
      lines.push(
        `${files.length - failures}/${files.length} test files passed` +
          (failures ? ` (${failures} failed)` : ""),
      );
      return {
        content: [{ type: "text", text: lines.join("\n") }],
        details: { ran: files.length, failures, ok: failures === 0 },
      };
    },
    renderCall(args, theme) {
      return new Text(
        theme.fg("toolTitle", theme.bold("test ")) + theme.fg("accent", args.file ?? "all"),
        0,
        0,
      );
    },
  });

  // ------------------------------------------------------------------- lint --
  pi.registerTool({
    name: "lint",
    label: "Ruff lint/format",
    description:
      "Run ruff format + ruff check via scripts/precommit.sh (which handles its own `uv run ruff` " +
      "and cd's to the repo root). action='fix' (default) auto-formats and applies safe fixes; " +
      "action='check' reports only (non-zero exit = issues found, no files changed).",
    promptSnippet: "Lint/format with ruff via scripts/precommit.sh (--check or fix)",
    parameters: Type.Object({
      action: Type.Optional(StringEnum(["fix", "check"] as const)),
    }),
    async execute(_id, params, signal, _onUpdate, ctx) {
      const root = findRoot(ctx.cwd);
      const check = (params.action ?? "fix") === "check";
      const r = await run(pi, "bash", ["scripts/precommit.sh", check ? "--check" : ""].filter(Boolean), {
        cwd: root,
        signal,
      });
      const ok = r.code === 0;
      const text =
        `$ bash scripts/precommit.sh ${check ? "--check" : ""}  [exit ${r.code}]` +
        ` — ${ok ? "clean" : "issues found"}\n${r.text}`;
      return {
        content: [{ type: "text", text }],
        details: { action: check ? "check" : "fix", code: r.code, ok },
      };
    },
    renderCall(args, theme) {
      return new Text(
        theme.fg("toolTitle", theme.bold("lint ")) + theme.fg("accent", String(args.action ?? "fix")),
        0,
        0,
      );
    },
  });

  // ---------------------------------------------------------------- build-c --
  pi.registerTool({
    name: "build-c",
    label: "Build C reference",
    description:
      "Build the C reference implementation (tausort.x) via make / g++. action='build' (default) " +
      "runs `make`; 'clean' runs `make clean`; 'rebuild' runs clean then build.",
    promptSnippet: "Build the C reference impl with make (tausort.x)",
    parameters: Type.Object({
      action: Type.Optional(StringEnum(["build", "clean", "rebuild"] as const)),
    }),
    async execute(_id, params, signal, _onUpdate, ctx) {
      const root = findRoot(ctx.cwd);
      const action = params.action ?? "build";
      let r;
      if (action === "clean") {
        r = await run(pi, "make", ["clean"], { cwd: root, signal });
      } else {
        if (action === "rebuild") await run(pi, "make", ["clean"], { cwd: root, signal });
        r = await run(pi, "make", [], { cwd: root, signal });
      }
      const ok = r.code === 0;
      const text = `$ make (${action})  [exit ${r.code}] — ${ok ? "ok" : "failed"}\n${r.text}`;
      return {
        content: [{ type: "text", text }],
        details: { action, code: r.code, ok },
      };
    },
    renderCall(args, theme) {
      return new Text(
        theme.fg("toolTitle", theme.bold("build-c ")) + theme.fg("accent", String(args.action ?? "build")),
        0,
        0,
      );
    },
  });

  // --------------------------------------------- bash spawn hook (force uv) --
  // Overrides the built-in `bash` tool. Only bare command-position invocations of
  // python/python3/pytest/ruff/basedpyright/pyright get an `uv run ` prefix; everything
  // else (make, jj, git, scripts, venv-path python, already-uv'd commands) is untouched.
  const defaultCwd = process.cwd();
  const bashTool = createBashTool(defaultCwd, {
    spawnHook: ({ command, cwd, env }) => ({
      command: withUvRun(command),
      cwd,
      env: { ...env, TAUSORT_UV_HOOK: "1" },
    }),
  });
  pi.registerTool({
    ...bashTool,
    promptGuidelines: [
      "Bare python/python3/pytest/ruff/basedpyright/pyright commands are run through `uv run` " +
        "automatically by the shell hook — invoke them directly without adding `uv run` yourself, " +
        "and do not wrap commands that already use uv/uvx or the ./.venv interpreter.",
    ],
  });
}

// --- command tokens that should be routed through `uv run` when invoked bare ----
const UV_TOKENS = new Set([
  "python",
  "python3",
  "pytest",
  "py.test",
  "ruff",
  "basedpyright",
  "pyright",
]);
// leading keywords that prefix a real command (skip them to find the command word)
const PASSTHROUGH = new Set(["time", "exec", "command", "nice", "nohup"]);

// Transform a single command segment (between && || ; |): if its command-position
// token is a bare python/pytest/ruff/etc., prefix with `uv run `.
function transformSegment(seg: string): string {
  const m = seg.match(
    /^(\s*)(env\s+)?((?:[A-Za-z_][A-Za-z0-9_]*=\S+\s+)*)((?:(?:time|exec|command|nice|nohup)\s+)?)([A-Za-z0-9_.\/:-]*)/,
  );
  if (!m) return seg;
  const [, ws, envKw, envAssigns, passKw, cmd] = m;
  if (!cmd) return seg; // subshells/$()/strings — leave untouched
  if (cmd === "uv" || cmd === "uvx") return seg; // already uv-managed
  if (cmd.includes("python") && /[\\/]/.test(cmd)) return seg; // e.g. .venv/bin/python
  if (!UV_TOKENS.has(cmd)) return seg;
  const head = (ws ?? "") + (envKw ?? "") + (envAssigns ?? "") + (passKw ?? "");
  return head + "uv run " + seg.slice(head.length);
}

// Rewrite top-level command segments of `command`, leaving separators intact.
function withUvRun(command: string): string {
  if (!command) return command;
  // if any segment already starts with `uv run` / `uvx`, skip it individually (per-segment)
  return command
    .split(/(&&|\|\||;|\|)/)
    .map((part, i) => (i % 2 === 1 ? part : transformSegment(part)))
    .join("");
}
