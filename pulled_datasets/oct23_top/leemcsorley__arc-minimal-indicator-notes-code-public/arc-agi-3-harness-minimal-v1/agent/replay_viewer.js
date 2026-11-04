(() => {
  // ARC replay viewer (harness v2.5-log). Reads the JSON in #arc-replay-data; older replays still work.
  const root = document.getElementById("arc-replay-root");
  const dataTag = document.getElementById("arc-replay-data");
  if (!root || !dataTag) return;

  let replay;
  try {
    replay = JSON.parse(dataTag.textContent || "{}");
  } catch (e) {
    root.textContent = `Failed to parse replay data: ${e}`;
    return;
  }

  const events = Array.isArray(replay.events) ? replay.events : [];
  if (!events.length) {
    root.textContent = "No events in replay.";
    return;
  }

  const shape = Array.isArray(replay.board_shape) ? replay.board_shape : [64, 64];
  const ROWS = Number(shape[0]) || 64;
  const COLS = Number(shape[1]) || 64;
  const colorMap = replay.color_map_hex || {};

  const gameId = String(replay.game_id || "unknown");
  const idParts = gameId.split("-", 2);
  const gameName = idParts[0] || gameId;
  const versionHash = idParts.length > 1 ? idParts[1] : "";

  document.title = `${gameName} — ARC Replay`;

  // ── Board parsing ──
  // Main boards: one CSV string per row. Extra boards (per-action boards, win frame, death board, pinned board):
  // one hex digit per cell ("hex_rows_v1"). Both decode here.

  const isHexRow = (t) => t.length >= 2 && !t.includes(",") && /^[0-9a-fA-F]+$/.test(t);

  const parseRow = (rowText) => {
    const t = String(rowText ?? "").trim();
    if (!t) return new Array(COLS).fill(0);
    let row;
    if (isHexRow(t)) {
      row = Array.from(t, ch => parseInt(ch, 16));
    } else {
      const parts = t.includes(",") ? t.split(",") : t.split(/\s+/);
      row = parts.map(p => { const n = Number(p.trim()); return Number.isFinite(n) ? n : 0; });
    }
    while (row.length < COLS) row.push(0);
    return row;
  };

  const normalizeBoard = (raw) => {
    if (!Array.isArray(raw) || !raw.length) return [];
    if (Array.isArray(raw[0])) {
      return raw.map(row =>
        Array.isArray(row)
          ? row.map(v => { const n = Number(v); return Number.isFinite(n) ? n : 0; })
          : new Array(COLS).fill(0)
      );
    }
    if (typeof raw[0] === "string") return raw.map(parseRow);
    return [];
  };

  const boardCache = new WeakMap();
  const decodeBoard = (raw) => {
    if (!Array.isArray(raw)) return null;
    if (boardCache.has(raw)) return boardCache.get(raw);
    const b = normalizeBoard(raw);
    if (!b.length) return null;
    while (b.length < ROWS) b.push(new Array(COLS).fill(0));
    boardCache.set(raw, b);
    return b;
  };

  for (const ev of events) {
    ev._board = decodeBoard(ev.board) || [];
    while (ev._board.length < ROWS) ev._board.push(new Array(COLS).fill(0));
  }

  // ── Notes / event log / store: carried forward (older v2.5 replays: notes rebuilt from the note tool calls) ──

  {
    let notes = {}, eventLog = [], store = null, haveNotesField = false, rects = null;
    for (const ev of events) {
      // indicator rectangles: events without the fields (trims, older reset events) keep the previous selection
      if (ev.remaining_actions_rects !== undefined || ev.remaining_actions_rect !== undefined || ev.kind === "turn") {
        rects = { remaining_actions_rects: ev.remaining_actions_rects, remaining_actions_rect: ev.remaining_actions_rect };
      }
      ev._rects = rects;
      if (ev.notes && typeof ev.notes === "object" && !Array.isArray(ev.notes)) {
        notes = { ...ev.notes }; haveNotesField = true;
        ev._noteOps = Array.isArray(ev.note_ops) ? ev.note_ops : [];
      } else {
        const ops = [];
        if (!haveNotesField && Array.isArray(ev.llm_commands)) {
          const rejected = /\[Notes\] Not saved/.test(String(ev.harness_feedback || ""));
          for (const c of ev.llm_commands) {
            if (c.kind !== "note") continue;
            let d = null;
            try { d = JSON.parse(c.content); } catch (e) { d = null; }
            if (!d) continue;
            const k = String(d.key || "").trim(), t = String(d.text || "").trim();
            if (!k) continue;
            if (!t) { if (k in notes) { ops.push({ key: k, op: "deleted", old: notes[k] }); delete notes[k]; } continue; }
            if (rejected) { ops.push({ key: k, op: "rejected", text: t }); continue; }
            const old = notes[k];
            notes[k] = t;
            ops.push(old === undefined ? { key: k, op: "added" } : { key: k, op: "updated", old });
          }
        }
        ev._noteOps = ops;
      }
      ev._notes = { ...notes };
      if (Array.isArray(ev.event_log)) eventLog = ev.event_log;
      ev._eventLog = eventLog;
      if (ev.store_summary !== undefined) store = ev.store_summary;
      ev._store = store;
    }
  }

  // ── Slider markers: level completions (full height), game overs / resets (upper half), context trims (lower half) ──

  const levelTransitions = [];
  for (let i = 1; i < events.length; i++) {
    const prev = Number(events[i - 1].levels_completed) || 0;
    const curr = Number(events[i].levels_completed) || 0;
    if (curr > prev) levelTransitions.push(i);
  }
  const kindMarkers = [];
  events.forEach((ev, i) => {
    const k = String(ev.kind || "");
    if (k === "game_over_reset" || k === "game_over") kindMarkers.push({ i, cls: "go-marker" });
    else if (k === "level_reset") kindMarkers.push({ i, cls: "rs-marker" });
    else if (k === "compaction") kindMarkers.push({ i, cls: "tr-marker" });
  });

  // ── Utilities ──

  const esc = (s) => String(s ?? "")
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

  const fmt = (n) => {
    const v = Number(n);
    return Number.isFinite(v) ? Math.round(v).toLocaleString("en-US") : "0";
  };
  const fmtK = (n) => {
    const v = Number(n);
    if (!Number.isFinite(v)) return "?";
    return v >= 1000 ? `${(v / 1000).toFixed(v >= 100000 ? 0 : 1)}k` : `${Math.round(v)}`;
  };
  const fmtMin = (s) => {
    const v = Number(s);
    if (!Number.isFinite(v)) return "";
    return v >= 3600 ? `${Math.floor(v / 3600)}h${String(Math.floor((v % 3600) / 60)).padStart(2, "0")}` : `${Math.floor(v / 60)}m${String(Math.floor(v % 60)).padStart(2, "0")}`;
  };

  const scoreColor = (score) => {
    if (score < 5) return "#F93C31";
    if (score < 10) return "#FF851B";
    if (score < 25) return "#FFDC00";
    if (score < 50) return "#c6e04a";
    if (score < 75) return "#7EE85D";
    return "#4FCC30";
  };

  const normalizeColor = (hex) => {
    const c = String(hex || "").replace("#", "");
    return c.length >= 6 ? `#${c.slice(0, 6)}` : "#555555";
  };

  const toRgb = (hex) => {
    const c = normalizeColor(hex).replace("#", "");
    return [parseInt(c.slice(0, 2), 16), parseInt(c.slice(2, 4), 16), parseInt(c.slice(4, 6), 16)];
  };

  const lumVal = (r, g, b) => 0.299 * r + 0.587 * g + 0.114 * b;

  const rectsOf = (ev) => {
    const src = ev._rects || ev;
    if (Array.isArray(src.remaining_actions_rects) && src.remaining_actions_rects.length) {
      return src.remaining_actions_rects.filter(r => Array.isArray(r) && r.length === 4);
    }
    const r = src.remaining_actions_rect;
    return (Array.isArray(r) && r.length === 4) ? [r] : [];
  };

  const turnsText = (a, b) => (a == null || b == null || b < a) ? "no earlier turns" : (a === b ? `turn ${a}` : `turns ${a}–${b}`);

  const turnLabelOf = (ev, idx) => {
    const kind = String(ev.kind || "turn");
    if (kind === "initial") return "Start";
    if (kind === "turn") return ev.turn ? `Turn ${ev.turn}` : (ev.title || `Turn ${idx}`);
    if (kind === "game_over_reset") return "Game over";
    if (kind === "level_reset") return "RESET";
    if (kind === "compaction") return "Context trimmed";
    return (ev.title || kind.replace(/_/g, " "));
  };

  // ── Message parsing (old + new event formats) ──

  const parseMessages = (ev) => {
    const out = { summary: "", reasoning: "", llmOutput: "", harness: "" };

    if (ev.llm_reasoning !== undefined || ev.harness_feedback !== undefined) {
      out.summary = String(ev.llm_reasoning || "");
      out.reasoning = String(ev.llm_reasoning_content || "");
      const cmdsFull = Array.isArray(ev.llm_commands)
        ? ev.llm_commands.map(c => `── ${c.kind} ──\n${c.content}`).join("\n\n")
        : String(ev.llm_commands_summary || "");
      out.llmOutput = cmdsFull;
      out.harness = String(ev.harness_feedback || "");
    } else {
      const text = String(ev.message_text || "");
      const ri = text.indexOf("Reasoning:\n");
      const fi = text.indexOf("\n\nFeedback:\n");
      if (ri >= 0 && fi > ri) {
        out.summary = text.slice(ri + "Reasoning:\n".length, fi);
        out.harness = text.slice(fi + "\n\nFeedback:\n".length);
      } else if (ri >= 0) {
        out.summary = text.slice(ri + "Reasoning:\n".length);
      } else {
        out.harness = text;
      }
    }

    // System prompt for initial step
    if (ev.kind === "initial" && ev.system_prompt) {
      out.harness = String(ev.system_prompt);
    }

    return out;
  };

  // ── Line diff (LCS-based) ──

  const lineDiff = (oldText, newText) => {
    if (!oldText && !newText) return [];
    const a = (oldText || "").split("\n");
    const b = (newText || "").split("\n");
    if (!oldText) return b.map(l => ({ type: "add", text: l }));
    if (!newText) return a.map(l => ({ type: "del", text: l }));
    const m = a.length, n = b.length;
    const dp = [];
    for (let i = 0; i <= m; i++) dp[i] = new Uint16Array(n + 1);
    for (let i = 1; i <= m; i++)
      for (let j = 1; j <= n; j++)
        dp[i][j] = a[i - 1] === b[j - 1] ? dp[i - 1][j - 1] + 1 : Math.max(dp[i - 1][j], dp[i][j - 1]);
    const result = [];
    let i = m, j = n;
    while (i > 0 || j > 0) {
      if (i > 0 && j > 0 && a[i - 1] === b[j - 1]) { result.unshift({ type: "same", text: b[j - 1] }); i--; j--; }
      else if (j > 0 && (i === 0 || dp[i][j - 1] >= dp[i - 1][j])) { result.unshift({ type: "add", text: b[j - 1] }); j--; }
      else { result.unshift({ type: "del", text: a[i - 1] }); i--; }
    }
    return result;
  };

  // ── Python syntax highlighting ──

  const highlightPy = (code) => {
    let h = esc(code);
    const toks = [];
    // Phase 1: extract comments & strings into placeholders (no HTML inserted yet)
    h = h.replace(/#.*|&quot;(?:(?!&quot;)[^\\]|\\.)*&quot;|'(?:[^'\\]|\\.)*'/g, (m) => {
      const i = toks.length;
      toks.push(m[0] === '#' ? `<span class="c-com">${m}</span>` : `<span class="c-str">${m}</span>`);
      return `\x00T${i}\x01`;
    });
    // Phase 2: keywords, numbers, decorators (runs on clean text, no HTML to interfere)
    h = h.replace(
      /\b(def|return|if|elif|else|for|while|in|and|or|not|None|True|False|class|try|except|raise|pass|continue|break|with|as|import|from|yield|lambda|is)\b/g,
      '<span class="c-kw">$&</span>');
    h = h.replace(/\b(\d+(?:\.\d+)?)\b/g, '<span class="c-num">$&</span>');
    h = h.replace(/^(\s*@[\w.]+)/gm, '<span class="c-dec">$1</span>');
    // Phase 3: restore placeholders with their HTML spans
    h = h.replace(/\x00T(\d+)\x01/g, (_, i) => toks[Number(i)]);
    return h;
  };

  // ── CSS ──

  const style = document.createElement("style");
  style.textContent = `
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap');

    :root {
      --bg: #090c10;
      --panel: #0e1219;
      --panel2: #131920;
      --border: #1a2230;
      --border2: #242e3c;
      --text: #d8e0ea;
      --text2: #95a0b0;
      --muted: #5e6a7a;
      --muted2: #3a4452;
      --magenta: #E53AA3;
      --red: #F93C31;
      --blue: #1E93FF;
      --cyan: #67dcff;
      --yellow: #FFDC00;
      --orange: #FF851B;
      --green: #4FCC30;
      --purple: #A356D6;
      --turquoise: #2dd4bf;
      --violet: #b39ddb;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    * { scrollbar-width: thin; scrollbar-color: #2a3544 #0e1318; }
    *::-webkit-scrollbar { width: 8px; height: 8px; }
    *::-webkit-scrollbar-track { background: #0e1318; border-radius: 4px; }
    *::-webkit-scrollbar-thumb { background: #2a3544; border-radius: 4px; }
    *::-webkit-scrollbar-thumb:hover { background: #3a4b60; }

    body {
      margin: 0; background: var(--bg); color: var(--text);
      font-family: 'Inter', system-ui, sans-serif;
      font-size: 13px; line-height: 1.45;
      -webkit-font-smoothing: antialiased;
      background:
        radial-gradient(ellipse 1100px 500px at 82% -6%, rgba(229,58,163,0.07), transparent 55%),
        radial-gradient(ellipse 900px 450px at 8% 2%, rgba(30,147,255,0.05), transparent 50%),
        var(--bg);
    }

    #arc-replay-root {
      min-height: 100vh; display: grid;
      grid-template-rows: auto auto minmax(0, 1fr);
      gap: 8px; padding: 10px 12px; max-width: 1920px; margin: 0 auto;
    }

    /* ── Header ── */
    .hdr {
      border: 1px solid var(--border); border-radius: 10px; background: var(--panel);
      padding: 10px 16px; display: flex; align-items: center; justify-content: space-between;
      flex-wrap: wrap; gap: 8px 16px; box-shadow: 0 6px 24px rgba(0,0,0,0.25);
    }
    .hdr-left, .hdr-right { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
    .hdr-center { display: flex; align-items: baseline; gap: 6px; font-family: 'Space Grotesk', sans-serif; }
    .game-name { font-size: 22px; font-weight: 700; letter-spacing: -0.5px; color: #fff; text-shadow: 0 0 20px rgba(229,58,163,0.3); }
    .version { font-size: 12px; font-weight: 400; font-style: italic; color: var(--muted); }
    .score-val { font-family: 'JetBrains Mono', monospace; font-size: 15px; font-weight: 700; position: relative; cursor: default; }
    .chip {
      display: inline-flex; align-items: center; gap: 5px;
      border: 1px solid var(--border2); border-radius: 6px; padding: 3px 9px;
      font-size: 11px; color: var(--text2); background: rgba(255,255,255,0.015); white-space: nowrap;
    }
    .chip b { color: var(--text); font-weight: 600; font-family: 'JetBrains Mono', monospace; font-size: 10.5px; }
    .hover-tip {
      display: none; position: absolute; top: calc(100% + 6px); left: 50%; transform: translateX(-50%);
      min-width: 300px; max-width: 560px; max-height: 70vh; overflow: auto;
      background: var(--panel2); border: 1px solid var(--border2); border-radius: 8px;
      padding: 8px 10px; font-family: 'JetBrains Mono', monospace; font-size: 10.5px; font-weight: 400; line-height: 1.45;
      color: var(--text); z-index: 200; box-shadow: 0 8px 28px rgba(0,0,0,0.5); white-space: normal; text-align: left;
    }
    .score-val:hover .hover-tip { display: block; }
    .hover-tip table { border-collapse: collapse; width: 100%; margin-top: 4px; }
    .hover-tip th, .hover-tip td { padding: 2px 6px; text-align: right; border-bottom: 1px solid var(--border); }
    .hover-tip th { color: var(--text2); font-weight: 600; }
    .hover-tip .tip-note { color: var(--muted); margin-top: 6px; font-size: 9.5px; }

    /* ── Slider ── */
    .slider-bar {
      border: 1px solid var(--border); border-radius: 10px; background: var(--panel);
      padding: 8px 12px; display: flex; align-items: center; gap: 8px;
      box-shadow: 0 4px 16px rgba(0,0,0,0.2);
    }
    .nav-btn {
      background: var(--panel2); border: 1px solid var(--border2); color: var(--text);
      border-radius: 6px; width: 32px; height: 28px; cursor: pointer;
      font-size: 12px; font-weight: 700; display: flex; align-items: center; justify-content: center;
    }
    .nav-btn:hover:not(:disabled) { background: var(--border2); }
    .nav-btn:disabled { opacity: 0.35; cursor: default; }
    .slider-wrap { flex: 1; position: relative; height: 28px; display: flex; align-items: center; }
    .slider-wrap input[type=range] {
      -webkit-appearance: none; appearance: none; width: 100%; height: 28px;
      background: transparent; cursor: pointer; position: relative; z-index: 3;
    }
    .slider-wrap input[type=range]::-webkit-slider-runnable-track { background: var(--border); height: 5px; border-radius: 3px; }
    .slider-wrap input[type=range]::-webkit-slider-thumb {
      -webkit-appearance: none; width: 15px; height: 15px; border-radius: 50%;
      background: var(--magenta); border: 2px solid #fff; margin-top: -5px;
      box-shadow: 0 0 10px rgba(229,58,163,0.5); box-sizing: border-box;
    }
    .slider-wrap input[type=range]::-moz-range-track { background: var(--border); height: 5px; border-radius: 3px; border: none; }
    .slider-wrap input[type=range]::-moz-range-thumb {
      width: 15px; height: 15px; border-radius: 50%;
      background: var(--magenta); border: 2px solid #fff;
      box-shadow: 0 0 10px rgba(229,58,163,0.5); box-sizing: border-box;
    }
    .lvl-marker {
      position: absolute; top: 3px; bottom: 3px; width: 3px;
      background: var(--yellow); border-radius: 1px; z-index: 2; pointer-events: none;
      box-shadow: 0 0 8px rgba(255,220,0,0.45);
    }
    .lvl-marker::after {
      content: attr(data-label); position: absolute; top: -14px; left: 50%;
      transform: translateX(-50%); font-size: 8px; font-weight: 700; color: var(--yellow);
      font-family: 'JetBrains Mono', monospace; white-space: nowrap;
    }
    /* game over (red) and RESET (orange): from the top down to the track line, half as tall as a level marker */
    .go-marker, .rs-marker {
      position: absolute; top: 3px; height: 11px; width: 3px; border-radius: 1px; z-index: 2; pointer-events: none;
    }
    .go-marker { background: var(--red); box-shadow: 0 0 6px rgba(249,60,49,0.5); }
    .rs-marker { background: var(--orange); box-shadow: 0 0 6px rgba(255,133,27,0.5); }
    /* context trim: a short grey tick below the track line */
    .tr-marker {
      position: absolute; top: 15px; height: 8px; width: 2px; border-radius: 1px; z-index: 2; pointer-events: none;
      background: #7f8da0; opacity: 0.8;
    }
    .turn-text {
      color: var(--text2); font-family: 'JetBrains Mono', monospace;
      font-size: 11px; font-weight: 500; white-space: nowrap; min-width: 190px; text-align: right;
    }

    /* ── Main layout ── */
    .main { display: grid; grid-template-columns: 1fr 2fr; gap: 8px; min-height: 0; height: calc(100vh - 120px); }

    /* ── Left column ── */
    .left-col { display: grid; grid-template-rows: minmax(0, 3fr) 26px minmax(0, 2fr); gap: 6px; min-height: 0; }
    .board-wrap {
      border: 1px solid var(--border); border-radius: 8px; background: #070a0e;
      padding: 8px; overflow: hidden; min-height: 0; display: grid; place-items: center; position: relative;
    }
    .board-label {
      position: absolute; top: 6px; left: 10px; font-family: 'JetBrains Mono', monospace; font-size: 10px;
      color: var(--text2); background: rgba(9,12,16,0.8); padding: 1px 6px; border-radius: 4px; pointer-events: none; z-index: 2;
    }
    .board-label:empty { display: none; }
    canvas { display: block; image-rendering: crisp-edges; image-rendering: pixelated; border-radius: 4px; }

    /* Step bar: one chip per executed action (click = board after it), plus badges */
    .fn-bar {
      border: 1px solid var(--border); border-radius: 6px; background: var(--panel);
      padding: 0 6px; display: flex; align-items: center; gap: 6px; min-width: 0;
      font-family: 'JetBrains Mono', monospace; font-size: 9px;
    }
    .chips { position: relative; flex: 1; min-width: 0; display: flex; align-items: center; gap: 3px; overflow-x: auto; overflow-y: hidden; white-space: nowrap; height: 100%; scrollbar-width: none; }
    .chips::-webkit-scrollbar { display: none; }
    .chips-empty { color: var(--muted2); font-style: italic; }
    .vchip {
      flex: none; padding: 1px 5px; border-radius: 4px; cursor: pointer; border: 1px solid var(--border2);
      background: var(--panel2); color: var(--text2); line-height: 14px; user-select: none;
    }
    .vchip:hover { border-color: #3a4b60; color: var(--text); }
    .vchip.sel { background: rgba(229,58,163,0.18); border-color: var(--magenta); color: #fff; }
    .vchip.v-noop { color: var(--muted); }
    .vchip.v-death { border-color: rgba(249,60,49,0.6); color: var(--red); }
    .vchip.v-win { border-color: rgba(79,204,48,0.6); color: var(--green); }
    .vchip.v-frame { border-style: dashed; border-color: rgba(79,204,48,0.6); color: var(--green); }
    .vchip.v-pin { border-color: rgba(127,141,160,0.7); color: #b8c3d2; }
    .vchip.v-end { color: var(--text); }
    .badges { flex: none; display: flex; align-items: center; gap: 4px; }
    .fn-tag {
      padding: 1px 6px; border-radius: 4px; font-weight: 700; letter-spacing: 0.2px;
      cursor: default; position: relative; line-height: 14px;
    }
    .fn-yes { background: rgba(79,204,48,0.12); color: var(--green); border: 1px solid rgba(79,204,48,0.25); }
    .fn-no { background: rgba(249,60,49,0.06); color: #9a4040; border: 1px solid rgba(249,60,49,0.15); }
    .fn-info { background: rgba(127,141,160,0.08); color: #9aa6b6; border: 1px solid rgba(127,141,160,0.25); }
    .fn-tag .sym-tip {
      display: none; position: absolute; bottom: calc(100% + 6px); right: 0;
      min-width: 280px; max-width: 520px; max-height: 420px; overflow: auto;
      background: var(--panel2); border: 1px solid var(--border2); border-radius: 8px;
      padding: 8px 10px; font-size: 10px; font-weight: 400; line-height: 1.4;
      white-space: pre-wrap; word-break: break-word; z-index: 100;
      box-shadow: 0 8px 28px rgba(0,0,0,0.5); color: var(--text);
    }
    .fn-tag:hover .sym-tip { display: block; }

    /* Code card */
    .code-card {
      border: 1px solid var(--border); border-top: 2px solid var(--blue); border-radius: 8px;
      background: var(--panel); display: grid; grid-template-rows: auto minmax(0, 1fr); min-height: 0; overflow: hidden;
    }
    .panel-hdr {
      display: flex; align-items: center; justify-content: space-between;
      padding: 6px 10px; border-bottom: 1px solid var(--border);
      font-family: 'JetBrains Mono', monospace; font-size: 10px; font-weight: 600;
      letter-spacing: 0.5px; text-transform: uppercase; color: var(--text2);
    }
    .seg { display: inline-flex; gap: 2px; }
    .seg button {
      background: transparent; border: 1px solid var(--border2); color: var(--muted); cursor: pointer;
      font-family: 'JetBrains Mono', monospace; font-size: 9px; font-weight: 700; letter-spacing: 0.3px;
      padding: 1px 7px; border-radius: 4px; text-transform: uppercase;
    }
    .seg button.on { color: var(--yellow); border-color: rgba(255,220,0,0.45); background: rgba(255,220,0,0.07); }
    .code-badge { font-size: 9px; font-weight: 700; padding: 1px 6px; border-radius: 4px; letter-spacing: 0.3px; }
    .code-badge.modified { background: rgba(79,204,48,0.15); color: var(--green); border: 1px solid rgba(79,204,48,0.3); }
    .code-body {
      overflow: auto; padding: 0; margin: 0; font-family: 'JetBrains Mono', monospace;
      font-size: 11px; line-height: 1.5; tab-size: 2; min-height: 0;
    }
    .code-line { display: flex; min-height: 17px; padding: 0 8px 0 0; }
    .ln { width: 36px; text-align: right; padding-right: 8px; color: var(--muted2); user-select: none; flex-shrink: 0; font-size: 10px; line-height: 17px; }
    .lc { flex: 1; white-space: pre; min-width: 0; }
    .line-add { background: rgba(79,204,48,0.07); }
    .line-add .ln { color: var(--green); }
    .line-del { background: rgba(249,60,49,0.07); text-decoration: line-through; opacity: 0.6; }
    .line-del .ln { color: var(--red); }
    .code-empty { padding: 16px; color: var(--muted); font-style: italic; font-size: 11px; }
    .c-kw { color: #FF7B72; } .c-num { color: #FFD166; } .c-com { color: #6A9955; } .c-str { color: #A5D6FF; } .c-dec { color: #D2A8FF; }

    /* ── Right column: 2×2 grid ── */
    .right-col {
      display: grid;
      grid-template-columns: 1fr 1fr;
      grid-template-rows: auto minmax(0, 1fr) minmax(0, 1fr);
      gap: 8px; min-height: 0;
    }
    .info-bar {
      grid-column: 1 / -1;
      display: flex; align-items: center; gap: 6px; flex-wrap: wrap;
      padding: 5px 10px; border: 1px solid var(--border); border-radius: 8px;
      background: var(--panel); font-family: 'JetBrains Mono', monospace; font-size: 11px;
    }
    .info-tag {
      padding: 2px 7px; border-radius: 4px; font-size: 10px; font-weight: 600;
      text-transform: uppercase; letter-spacing: 0.3px;
    }
    .info-tag.kind-initial { background: rgba(30,147,255,0.12); color: var(--blue); border: 1px solid rgba(30,147,255,0.25); }
    .info-tag.kind-turn { background: rgba(229,58,163,0.1); color: var(--magenta); border: 1px solid rgba(229,58,163,0.2); }
    .info-tag.kind-success, .info-tag.kind-level_complete { background: rgba(79,204,48,0.12); color: var(--green); border: 1px solid rgba(79,204,48,0.25); }
    .info-tag.kind-game_over, .info-tag.kind-game_over_reset, .info-tag.kind-llm_error, .info-tag.kind-length_cap { background: rgba(249,60,49,0.12); color: var(--red); border: 1px solid rgba(249,60,49,0.25); }
    .info-tag.kind-max_turns, .info-tag.kind-level_reset, .info-tag.kind-deadline { background: rgba(255,133,27,0.12); color: var(--orange); border: 1px solid rgba(255,133,27,0.25); }
    .info-tag.kind-compaction, .info-tag.kind-guard { background: rgba(127,141,160,0.1); color: #a9b6c7; border: 1px solid rgba(127,141,160,0.3); }
    .info-sep { color: var(--muted2); } .info-val { color: var(--text2); } .info-val b { color: var(--text); font-weight: 600; }
    .info-val.dim { color: var(--muted); }

    .msg-card {
      border: 1px solid var(--border); border-radius: 8px; background: var(--panel);
      display: grid; grid-template-rows: auto minmax(0, 1fr); min-height: 0; overflow: hidden;
    }
    .msg-card.llm-out   { border-top: 2px solid var(--green);     grid-column: 1; grid-row: 2; }
    .msg-card.harness    { border-top: 2px solid var(--yellow);    grid-column: 1; grid-row: 3; }
    .msg-card.summary    { border-top: 2px solid var(--turquoise); grid-column: 2; grid-row: 2; }
    .msg-card.reasoning  { border-top: 2px solid var(--violet);    grid-column: 2; grid-row: 3; }

    .msg-body {
      overflow: auto; padding: 8px 10px; margin: 0; white-space: pre-wrap; word-break: break-word;
      font-family: 'JetBrains Mono', monospace; font-size: 11.5px; line-height: 1.45;
      color: var(--text); min-height: 0;
    }
    .msg-empty { color: var(--muted); font-style: italic; }

    /* Pinned notes / event log, shown under the summary (greyed: carried state, not this turn's text) */
    .pin-sec { margin-top: 12px; padding-top: 6px; border-top: 1px dashed var(--border2); color: #8793a3; font-size: 10.5px; }
    .pin-hdr { font-size: 9.5px; font-weight: 700; letter-spacing: 0.5px; text-transform: uppercase; color: #6f7c8d; margin-bottom: 3px; }
    .pin-hdr span { font-weight: 400; text-transform: none; letter-spacing: 0; }
    .note-row { margin: 2px 0 4px; }
    .note-key { color: #a9b6c7; font-weight: 700; }
    .note-op { font-size: 9px; font-weight: 700; padding: 0 4px; border-radius: 3px; margin-right: 4px; }
    .op-added { color: var(--green); border: 1px solid rgba(79,204,48,0.4); }
    .op-updated { color: var(--yellow); border: 1px solid rgba(255,220,0,0.35); }
    .op-deleted, .op-rejected { color: var(--red); border: 1px solid rgba(249,60,49,0.4); }
    .note-row.deleted { text-decoration: line-through; opacity: 0.6; }
    .note-row.changed { color: #b9c4d2; }
    .elog-row { margin: 1px 0; }
    .elog-row.go { color: #c9726c; } .elog-row.rs { color: #c9925c; } .elog-row.ok { color: #79b56a; }

    /* Model-input view sections */
    .in-sec {
      margin: 10px 0 4px; padding: 2px 6px; border-left: 2px solid var(--yellow); background: rgba(255,220,0,0.04);
      font-size: 9.5px; font-weight: 700; letter-spacing: 0.4px; text-transform: uppercase; color: var(--text2);
    }
    .in-sec:first-child { margin-top: 0; }
    .in-sec span { font-weight: 400; text-transform: none; letter-spacing: 0; color: var(--muted); }
    .in-ctx { color: var(--text2); font-size: 10.5px; }

    /* Inline images in messages */
    .msg-body img.replay-img {
      display: block; margin: 6px 0;
      border: 1px solid var(--border2); border-radius: 6px;
      image-rendering: crisp-edges; image-rendering: pixelated;
    }
    .msg-body img.replay-img-big { max-width: 100%; }
    .msg-body img.replay-img-small { max-width: 25%; max-height: 128px; }

    /* Collapsible sections */
    .collapsible-hdr {
      cursor: pointer; padding: 4px 8px; margin: 4px 0 2px;
      background: var(--panel2); border: 1px solid var(--border); border-radius: 4px;
      font-size: 10px; font-weight: 600; text-transform: uppercase; color: var(--text2);
      user-select: none; display: flex; align-items: center; gap: 4px;
    }
    .collapsible-hdr::before { content: '▶'; font-size: 8px; transition: transform 0.15s; }
    .collapsible-hdr.open::before { transform: rotate(90deg); }
    .collapsible-body { display: none; margin: 2px 0 6px; padding: 6px 8px; background: rgba(0,0,0,0.15); border-radius: 4px; font-size: 10px; overflow: auto; max-height: 400px; }
    .collapsible-body.open { display: block; }

    /* ── Responsive ── */
    @media (max-width: 1200px) {
      .main { grid-template-columns: 1fr; height: auto; }
      .left-col { grid-template-rows: minmax(0, 3fr) 26px minmax(0, 2fr); }
      .right-col { grid-template-columns: 1fr 1fr; }
      .board-wrap { min-height: 300px; max-height: 65vh; }
    }
    @media (max-width: 700px) {
      #arc-replay-root { padding: 8px; gap: 6px; }
      .hdr { flex-direction: column; align-items: flex-start; }
      .hdr-center { order: -1; }
      .right-col { grid-template-columns: 1fr; grid-template-rows: auto auto auto auto auto; }
      .msg-card.llm-out   { grid-column: 1; grid-row: 2; }
      .msg-card.harness    { grid-column: 1; grid-row: 3; }
      .msg-card.summary    { grid-column: 1; grid-row: 4; }
      .msg-card.reasoning  { grid-column: 1; grid-row: 5; }
    }
  `;
  document.head.appendChild(style);

  // ── Header HTML ──

  const lastEv = events[events.length - 1];
  const levelsDone = Number(lastEv.levels_completed) || 0;
  const levelScores = Array.isArray(replay.level_scores) ? replay.level_scores
    : (Array.isArray(lastEv.level_scores) ? lastEv.level_scores : null);
  // Replays before v2.5-log took the score from the shared scorecard, which was empty in the public test (0.0).
  const scoreUnreliable = !replay.score_source && Number(replay.scorecard_total_levels) === 0 && levelsDone > 0;
  const scoreValue = (() => {
    if (scoreUnreliable) return null;
    const n = Number(replay.final_score);
    return (replay.final_score !== null && replay.final_score !== undefined && Number.isFinite(n)) ? n : null;
  })();
  const tokenTotals = (replay.token_usage_totals && typeof replay.token_usage_totals === "object") ? replay.token_usage_totals : null;
  const tokenMiss = tokenTotals ? Number(tokenTotals.input_cache_miss_tokens ?? tokenTotals.input_miss_tokens ?? 0) : 0;
  const tokenCached = tokenTotals ? Number(tokenTotals.input_cached_tokens ?? 0) : 0;
  const tokenOutput = tokenTotals ? Number(tokenTotals.output_tokens ?? 0) : 0;

  const scoreTipHtml = (() => {
    let h = "";
    if (scoreUnreliable) {
      h += `<div>No per-game score in this replay: it was written by an older harness that read the shared scorecard, which was empty in the in-process public test (it showed 0.0 for every game).</div>`;
    } else if (replay.score_source) {
      h += `<div>Score source: ${esc(replay.score_source)}</div>`;
    }
    if (levelScores && levelScores.length) {
      h += `<table><tr><th>Level</th><th>Actions</th><th>Baseline</th><th>Score</th><th>Weight</th></tr>`;
      for (const r of levelScores) {
        h += `<tr><td>${esc(r.level)}</td><td>${r.actions == null ? "–" : esc(r.actions)}</td><td>${r.baseline == null ? "?" : esc(r.baseline)}</td>`
          + `<td>${r.score == null ? "?" : Number(r.score).toFixed(1)}</td><td>${esc(r.level)}</td></tr>`;
      }
      h += `</table>`;
      h += `<div class="tip-note">Level score = min(115, 100·(baseline/actions)²); game score = level-weighted mean. Actions per level count every env step on that level, including RESETs and the automatic reset after a game over.</div>`;
    }
    return h;
  })();

  const scoreHtml = scoreValue !== null
    ? `<span class="score-val" style="color:${scoreColor(scoreValue)}">${scoreValue.toFixed(1)}${scoreTipHtml ? `<div class="hover-tip">${scoreTipHtml}</div>` : ""}</span>`
    : `<span class="score-val" style="color:var(--muted)">n/a${scoreTipHtml ? `<div class="hover-tip">${scoreTipHtml}</div>` : ""}</span>`;
  const tokenChipHtml = tokenTotals
    ? `<span class="chip">Tokens <b>${fmt(tokenMiss)}</b> miss · <b>${fmt(tokenCached)}</b> cached · <b>${fmt(tokenOutput)}</b> out</span>` : "";
  const winLevels = Number(replay.win_levels) || (levelScores ? levelScores.length : 0);
  const levelsChipHtml = `<span class="chip">Levels <b>${levelsDone}${winLevels ? ` / ${winLevels}` : ""}</b></span>`;
  const harnessChipHtml = replay.harness_version ? `<span class="chip">Harness <b>${esc(replay.harness_version)}</b></span>` : "";
  // v2.6: layered tracker compute (worker CPU, per-board time) from the replay meta
  const trk = (replay.layered_tracker && typeof replay.layered_tracker === "object") ? replay.layered_tracker : null;
  const trackerChipHtml = (() => {
    if (!trk) return "";
    if (!trk.frames) return trk.off_reason ? `<span class="chip" title="${esc(trk.off_reason)}">Tracker <b>off</b></span>` : "";
    const tip = `Layered tracker: ${trk.frames} boards; worker CPU ${trk.cpu_s} s; per board mean ${trk.mean_ms} ms, `
      + `p90 ${trk.p90_ms} ms, max ${Math.round(Number(trk.max_ms) || 0)} ms; the game waited ${trk.waits_s} s in all`
      + (trk.skipped ? `; ${trk.skipped} board(s) skipped as too complex` : "")
      + (trk.restarts ? `; ${trk.restarts} restart(s)` : "")
      + (trk.off_reason ? `; stopped: ${trk.off_reason}` : "");
    return `<span class="chip" title="${esc(tip)}">Tracker <b>${esc(trk.cpu_s)} s</b> CPU · <b>${esc(trk.mean_ms)}</b> ms/board</span>`;
  })();

  // ── DOM ──

  root.innerHTML = `
    <div class="hdr">
      <div class="hdr-left">
        <span class="chip">Model <b>${esc(replay.model_name || "unknown")}</b></span>
        <span class="chip">Effort <b>${esc(replay.thinking_effort || "n/a")}</b></span>
        ${harnessChipHtml}
      </div>
      <div class="hdr-center">
        <span class="game-name">${esc(gameName)}</span>
        ${versionHash ? `<span class="version">(${esc(versionHash)})</span>` : ""}
        ${scoreHtml}
      </div>
      <div class="hdr-right">
        ${levelsChipHtml}
        ${tokenChipHtml}
        ${trackerChipHtml}
        <span class="chip">Start <b>${esc(replay.start_stamp || "n/a")}</b></span>
      </div>
    </div>
    <div class="slider-bar">
      <button class="nav-btn" id="prevBtn" title="Previous event (←)">◀</button>
      <div class="slider-wrap" id="sliderWrap">
        <input id="slider" type="range" min="0" max="${Math.max(events.length - 1, 0)}" value="${events.length - 1}" />
      </div>
      <button class="nav-btn" id="nextBtn" title="Next event (→)">▶</button>
      <span class="turn-text" id="turnText"></span>
    </div>
    <div class="main">
      <div class="left-col">
        <div class="board-wrap" id="boardWrap"><div class="board-label" id="boardLabel"></div><canvas id="boardCanvas"></canvas></div>
        <label style="padding:6px"><input type="checkbox" id="entityNames"> Entity names</label>
        <details style="padding:6px"><summary>Object references / legacy interpretation</summary><pre id="entityInspector" style="white-space:pre-wrap;max-height:280px;overflow:auto"></pre></details>
        <div class="fn-bar" id="fnBar"><div class="chips" id="chips"></div><div class="badges" id="badges"></div></div>
        <div class="code-card">
          <div class="panel-hdr"><span>Virtual .py File</span><span class="code-badge" id="codeBadge"></span></div>
          <div class="code-body" id="codeBody"></div>
        </div>
      </div>
      <div class="right-col">
        <div class="info-bar" id="infoBar"></div>
        <div class="msg-card llm-out">
          <div class="panel-hdr"><span>LLM Output</span></div>
          <pre class="msg-body" id="llmBody"></pre>
        </div>
        <div class="msg-card harness">
          <div class="panel-hdr"><span id="harnessTitle">Harness Response</span><span class="seg" id="harnessSeg">
            <button data-mode="result" title="What the harness returned for this turn">Result</button>
            <button data-mode="input" title="What the model was given for this turn (request text, tool result, images, context)">Model input</button>
          </span></div>
          <pre class="msg-body" id="harnessBody"></pre>
        </div>
        <div class="msg-card summary">
          <div class="panel-hdr"><span>Summary</span><span style="text-transform:none;font-weight:400;color:var(--muted)">+ pinned notes &amp; event log</span></div>
          <pre class="msg-body" id="summaryBody"></pre>
        </div>
        <div class="msg-card reasoning">
          <div class="panel-hdr"><span>Reasoning</span></div>
          <pre class="msg-body" id="reasoningBody"></pre>
        </div>
      </div>
    </div>
  `;

  // ── Element refs ──

  const $ = (id) => document.getElementById(id);
  const boardWrap = $("boardWrap");
  const boardLabel = $("boardLabel");
  const canvas = $("boardCanvas");
  const ctx = canvas.getContext("2d");
  const sliderWrap = $("sliderWrap");
  const slider = $("slider");
  const prevBtn = $("prevBtn");
  const nextBtn = $("nextBtn");
  const turnText = $("turnText");
  const infoBar = $("infoBar");
  const llmBody = $("llmBody");
  const harnessBody = $("harnessBody");
  const harnessTitle = $("harnessTitle");
  const harnessSeg = $("harnessSeg");
  const summaryBody = $("summaryBody");
  const reasoningBody = $("reasoningBody");
  const codeBody = $("codeBody");
  const codeBadge = $("codeBadge");
  const chipsEl = $("chips");
  const badgesEl = $("badges");

  // ── Slider markers ──

  const THUMB_W = 15;

  const placeMarkers = () => {
    sliderWrap.querySelectorAll(".lvl-marker, .go-marker, .rs-marker, .tr-marker").forEach(el => el.remove());
    const total = events.length - 1;
    if (total <= 0) return;
    // Browser positions the thumb center at: pct * (100% - THUMB_W) + THUMB_W/2; marker left = center - width/2
    const place = (pos, cls, width, label) => {
      const pct = Math.max(0, Math.min(1, pos / total));
      const pxOff = THUMB_W / 2 - pct * THUMB_W - width / 2;
      const mk = document.createElement("div");
      mk.className = cls;
      mk.style.left = `calc(${pct * 100}% + ${pxOff}px)`;
      if (label) mk.dataset.label = label;
      sliderWrap.appendChild(mk);
    };
    for (const m of kindMarkers) place(m.i, m.cls, m.cls === "tr-marker" ? 2 : 3, "");
    for (const ti of levelTransitions) place(ti - 0.5, "lvl-marker", 3, `L${Number(events[ti].levels_completed) || 1}`);
  };

  // ── Board views (the end-of-event board, and per action / win frame / game-over board when recorded) ──

  const coordsOfAction = (text) => {
    const m = /^\S+\s+(-?\d+)\s+(-?\d+)/.exec(String(text || ""));
    return m ? { row: Number(m[1]), col: Number(m[2]) } : null;
  };

  const stepTitle = (s) => {
    const parts = [`#${s.n ?? "?"} ${s.action}`];
    if (s.outcome && s.outcome !== "continue") parts.push(`→ ${s.outcome}`);
    if (s.changed_cells != null) {
      parts.push(`${s.changed_cells} cell(s) changed` + (s.changed_outside_indicator != null && s.changed_outside_indicator !== s.changed_cells
        ? ` (${s.changed_outside_indicator} outside the indicator)` : ""));
    }
    if (s.anim_frames) parts.push(`${s.anim_frames} animation frame(s)`);
    return parts.join(" · ");
  };

  const buildViews = (ev) => {
    const views = [];
    const kind = String(ev.kind || "");
    const endLabel = kind === "turn" ? "end of turn" : "board";
    if (kind === "game_over_reset" && ev.death_board) {
      views.push({ label: "game over board", board: decodeBoard(ev.death_board), cls: "v-death", title: "The board when the game over happened (before the automatic reset)" });
      views.push({ label: "restarted", board: ev._board, cls: "v-end", title: "The level's opening board after the reset" });
      return { views, def: 0 };
    }
    if (kind === "level_reset" && ev.before_board) {
      views.push({ label: "before RESET", board: decodeBoard(ev.before_board), cls: "", title: "The board when the model sent RESET" });
      views.push({ label: "restarted", board: ev._board, cls: "v-end", title: "The level's opening board after the RESET" });
      return { views, def: 1 };
    }
    if (kind === "compaction" && ev.pinned_board) {
      views.push({ label: "current board", board: ev._board, cls: "v-end", title: "The board at the time of the trim" });
      views.push({ label: "pinned level start", board: decodeBoard(ev.pinned_board), cls: "v-pin", title: "The board image sent with the pinned level-start message" });
      return { views, def: 0 };
    }
    const steps = (ev.v2 && Array.isArray(ev.v2.steps)) ? ev.v2.steps : [];
    const lcs = Array.isArray(ev.level_complete) ? ev.level_complete : [];
    let lcUsed = 0;
    steps.forEach((s, k) => {
      if (!s || !Array.isArray(s.board)) return;
      const oc = String(s.outcome || "continue");
      const isWin = oc === "level_complete" || oc === "win" || (Number(s.level_after) > Number(s.level_before));
      if (isWin) {
        const lc = lcs[lcUsed++];
        if (lc && Array.isArray(lc.win_frame)) {
          views.push({ label: `★ win frame L${lc.level}`, board: decodeBoard(lc.win_frame), cls: "v-frame",
            title: "First animation frame after the winning action that differs from the board before it (before the level switched)" });
        }
      }
      let cls = "";
      if (oc === "game_over") cls = "v-death";
      else if (isWin) cls = "v-win";
      else if (s.changed_outside_indicator === 0 || (s.changed_outside_indicator == null && s.changed_cells === 0)) cls = "v-noop";
      views.push({ label: `${s.n ?? k + 1} ${s.action}`, board: decodeBoard(s.board), cls, title: stepTitle(s), step: s });
    });
    if (views.length) views.push({ label: endLabel, board: ev._board, cls: "v-end", title: "The board at the end of this event (what the next request shows)" });
    return { views, def: Math.max(0, views.length - 1) };
  };

  // ── Board rendering ──

  let cellPx = 10;

  const computeCell = () => {
    if (!boardWrap) return;
    const w = boardWrap.clientWidth - 16;
    const h = boardWrap.clientHeight - 16;
    cellPx = Math.max(6, Math.min(16, Math.min(Math.floor(w / COLS), Math.floor(h / ROWS))));
  };

  const syncCanvas = () => {
    computeCell();
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    const cw = COLS * cellPx, ch = ROWS * cellPx;
    canvas.style.width = `${cw}px`;
    canvas.style.height = `${ch}px`;
    canvas.width = Math.floor(cw * dpr);
    canvas.height = Math.floor(ch * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.imageSmoothingEnabled = false;
  };

  const drawBoard = (ev, view) => {
    syncCanvas();
    const board = (view && view.board) || ev._board || [];
    if (!board.length) { ctx.clearRect(0, 0, canvas.width, canvas.height); return; }

    const showText = cellPx >= 10;
    const fontPx = Math.max(7, Math.floor(cellPx * 0.55));
    if (showText) {
      ctx.font = `600 ${fontPx}px "JetBrains Mono", monospace`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
    }

    for (let r = 0; r < ROWS; r++) {
      const row = board[r] || [];
      for (let c = 0; c < COLS; c++) {
        const val = (c < row.length && Number.isFinite(row[c])) ? row[c] : 0;
        const hex = normalizeColor(colorMap[String(val)] || "#555555");
        ctx.fillStyle = hex;
        ctx.fillRect(c * cellPx, r * cellPx, cellPx, cellPx);
        if (cellPx >= 8) {
          ctx.strokeStyle = "rgba(0,0,0,0.2)";
          ctx.lineWidth = 0.5;
          ctx.strokeRect(c * cellPx + 0.5, r * cellPx + 0.5, cellPx - 1, cellPx - 1);
        }
        if (showText) {
          const [cr, cg, cb] = toRgb(hex);
          const dark = lumVal(cr, cg, cb) < 140;
          ctx.fillStyle = dark ? "rgba(255,255,255,0.55)" : "rgba(0,0,0,0.4)";
          ctx.fillText(String(val), c * cellPx + cellPx / 2, r * cellPx + cellPx / 2 + 0.5);
        }
      }
    }

    // Only overlay an interpretation on its own observed frame, never animation/winning-frame guesses.
    const ip = view && view.step ? view.step.interpretation :
      (!view || view.cls === "v-end") ? ev.v2 && ev.v2.interpretation : null;
    const refs = view && view.step ? view.step.object_refs :
      (!view || view.cls === "v-end") ? ev.v2 && ev.v2.object_refs : null;
    const refSnap = view && view.step ? refs && refs.snapshot : refs;
    const snap = refSnap ? {...refSnap, status:"valid", entities:(refSnap.components || []).filter(e => !e.is_background)
      .map(e => ({...e, name:e.id}))} : view && view.step ? ip && ip.after : ip && ip.snapshot;
    const matches = snap && snap.status === "valid" && snap.shape[0] <= board.length &&
      snap.shape[1] <= (board[0] || []).length && (!view || view.cls === "v-end" || !!view.step) &&
      (snap.entities || []).every(e => e.cells.every(([r,c], i) => board[r] && board[r][c] === e.values[i]));
    $("entityInspector").textContent = refSnap ?
      `Object references | level ${refSnap.level}, attempt ${refSnap.attempt}, step ${refSnap.step}\n` +
      (refSnap.components || []).map(e => `${e.id}: ${e.size} cells, colours ${JSON.stringify(e.colors)}, rect ${JSON.stringify(e.rect)}`).join("\n") +
      `\n\nSaved names: ${JSON.stringify(refSnap.aliases || {})}\nEdges: ${JSON.stringify(refs.edges || [])}` +
      `\nChanges: ${JSON.stringify(refs.report || refs.changes || {})}\nChecks: ${JSON.stringify(refs.expectations || [])}` :
      !snap ? "No interpretation recorded for this frame." :
      `Status: ${snap.status} | level ${snap.level}, attempt ${snap.attempt}, step ${snap.step}\nRevision: ${String(snap.revision || '').slice(0,12)}\n` +
      (snap.error ? `Error: ${snap.error}\n` : "") +
      (snap.entities || []).map(e => `@${e.name} | claimed role: ${e.role}\nrect ${JSON.stringify(e.rect)}, centre ${JSON.stringify(e.centre)}, ${e.size} cells\nGrouping: ${e.grouping}\nEvidence: ${e.evidence}`).join("\n\n") +
      `\n\nChanges: ${JSON.stringify(ip && (ip.report || ip.changes) || {})}\nExpectations: ${JSON.stringify(ip && ip.expectations || [])}`;
    if (matches && $("entityNames").checked) {
      ctx.save();
      ctx.font = "bold 12px sans-serif";
      ctx.textAlign = "left"; ctx.textBaseline = "top";
      for (const e of snap.entities || []) {
        const [r0,r1,c0,c1] = e.rect;
        ctx.strokeStyle = "#00e5ff"; ctx.lineWidth = 2;
        ctx.strokeRect(c0*cellPx, r0*cellPx, (c1-c0)*cellPx, (r1-r0)*cellPx);
        const label = refSnap ? e.id : `@${e.name}`, width = ctx.measureText(label).width + 6;
        const x = Math.max(0, Math.min(c0*cellPx, COLS*cellPx-width));
        const y = Math.max(0, r0*cellPx-16);
        ctx.fillStyle = "rgba(0,0,0,.85)"; ctx.fillRect(x,y,width,16);
        ctx.fillStyle = "#00e5ff"; ctx.fillText(label,x+3,y+1);
      }
      ctx.restore();
    }

    // Remaining-actions indicator rectangles (dashed yellow, red if modified this turn)
    for (const rect of rectsOf(ev)) {
      const [rs, re, cs, ce] = rect.map(Number);
      ctx.strokeStyle = ev.remaining_actions_rect_modified ? "#F93C31" : "#FFDC00";
      ctx.lineWidth = 2.5;
      ctx.setLineDash([6, 3]);
      ctx.strokeRect(cs * cellPx, rs * cellPx, (ce - cs) * cellPx, (re - rs) * cellPx);
      ctx.setLineDash([]);
    }

    // Click cell highlights — proper border straddling cell edge
    let clicks = null;
    if (view && view.step) {
      const p = coordsOfAction(view.step.action);
      clicks = p ? [{ row: p.row, col: p.col, accepted: true }] : null;
    } else if (!view || view.cls === "v-end") {
      clicks = ev.click_cells;
    }
    if (Array.isArray(clicks)) {
      for (const click of clicks) {
        const cr = Number(click.row), cc = Number(click.col);
        if (cr >= 0 && cc >= 0 && cr < ROWS && cc < COLS) {
          ctx.strokeStyle = click.accepted ? "#4FCC30" : "#F93C31";
          ctx.lineWidth = 3;
          ctx.strokeRect(cc * cellPx, cr * cellPx, cellPx, cellPx);
        }
      }
    }
  };

  // ── Info bar ──

  const KNOWN_KINDS = ["initial", "turn", "success", "game_over", "game_over_reset", "level_reset", "compaction", "max_turns", "llm_error", "deadline"];
  const kindClass = (kind) => {
    const k = String(kind || "").toLowerCase().replace(/\s+/g, "_");
    return KNOWN_KINDS.includes(k) ? k : "turn";
  };

  const renderInfoBar = (ev) => {
    const kind = String(ev.kind || "turn");
    const acts = Number(ev.actions_taken) || 0;
    const lvlCompleted = Number(ev.levels_completed) || 0;
    const currentLvl = ev.current_level !== undefined ? Number(ev.current_level) : lvlCompleted + 1;
    const allDone = winLevels > 0 && lvlCompleted >= winLevels;
    const parts = [`<span class="info-tag kind-${kindClass(kind)}">${esc(kind.replace(/_/g, " "))}</span>`];
    const sep = `<span class="info-sep">·</span>`;
    const val = (h, cls = "") => `<span class="info-val ${cls}">${h}</span>`;
    parts.push(val(`Actions <b>${acts}</b>`));
    parts.push(sep, val(allDone ? `all <b>${winLevels}</b> levels done` : `Level <b>${currentLvl}</b>`));
    if (Number(ev.attempt) > 1 && !allDone) parts.push(val(`attempt <b>${ev.attempt}</b>`));
    if (ev.actions_on_level != null) parts.push(val(`(${ev.actions_on_level} on level)`, "dim"));
    parts.push(sep, val(`Completed <b>${lvlCompleted}</b>`));
    for (const lc of (Array.isArray(ev.level_complete) ? ev.level_complete : [])) {
      const sc = lc.level_score != null ? ` · ${Number(lc.level_score).toFixed(1)}` : "";
      const tip = `Level ${lc.level} ${lc.won_game ? "completed (game won)" : "completed"} after ${lc.actions_scored} scored action(s)`
        + (lc.baseline ? ` (baseline ${lc.baseline})` : "") + ` over ${lc.attempts} attempt(s); winning action: ${lc.winning_action || "?"}`;
      parts.push(`<span class="info-tag kind-level_complete" title="${esc(tip)}">✓ L${esc(lc.level)} ${esc(lc.actions_scored)}${lc.baseline ? "/" + esc(lc.baseline) : ""}${sc}</span>`);
    }
    if (ev.restart) {
      const r = ev.restart;
      const t = r.manual ? `RESET by the model after ${r.actions_in_ended_attempt} action(s)` : `game over after ${r.actions_in_ended_attempt} action(s)`;
      parts.push(sep, val(`${esc(t)} · attempt ${esc(r.ended_attempt)} → ${esc(r.new_attempt)} (${esc(r.where)})`));
    }
    if (ev.trim) {
      const t = ev.trim;
      parts.push(sep, val(`trim #${esc(t.n)} (${esc(t.why)}) ~${fmtK(t.est_tokens)} → ~${fmtK(t.kept_tokens)} tok · keeps `
        + turnsText(Array.isArray(t.kept_turns) ? t.kept_turns[0] : null, Array.isArray(t.kept_turns) ? t.kept_turns[1] : null)));
    }
    const u = ev.usage;
    if (u) {
      if (u.finish_reason === "length") parts.push(`<span class="info-tag kind-length_cap" title="The response hit the output-token cap and was cut off">length cap</span>`);
      parts.push(sep, val(`in <b>${fmtK(u.input_total)}</b> (${fmtK(u.input_cached)} cached) · out <b>${fmtK(u.output)}</b>`
        + (u.request_seconds != null ? ` · ${u.request_seconds}s` : "")
        + ((u.failed_attempts || u.rejected_attempts) ? ` · retries ${u.failed_attempts || 0}/${u.rejected_attempts || 0}` : "")));
    }
    const c = ev.context;
    if (c) parts.push(sep, val(`ctx: ${turnsText(c.first_turn, c.last_turn)}${c.pinned ? " + pinned" : ""}`, "dim"));
    const guards = (ev.v2 && Array.isArray(ev.v2.guards)) ? ev.v2.guards : [];
    if (guards.length) {
      const tip = guards.map(g => Object.entries(g).map(([k, v]) => `${k}=${typeof v === "object" ? JSON.stringify(v) : v}`).join(" ")).join("\n");
      parts.push(`<span class="info-tag kind-guard" title="${esc(tip)}">guards ${guards.length}</span>`);
    }
    if (ev.reasoning_control?.enabled) {
      const r = ev.reasoning_control;
      const tip = [r.question || "", r.trigger ? `Trigger: ${r.trigger}` : "",
        r.idle_turns != null ? `No-action responses: ${r.idle_turns}/${r.idle_turn_limit}; medium responses: ${r.medium_turns}/${r.medium_turn_limit}` : "",
        ...(r.events || []).map(e => e.question || e.reason || e.event)].filter(Boolean).join("\n");
      parts.push(`<span class="info-tag" title="${esc(tip)}">reasoning ${esc(r.mode)} · ${esc(ev.request_kind || r.kind)}</span>`);
    }
    if (ev.wall_s != null) parts.push(sep, val(`t=${fmtMin(ev.wall_s)}`, "dim"));
    infoBar.innerHTML = parts.join(" ");
  };

  // ── Step bar: view chips + badges ──

  let curViews = { views: [], def: 0 };
  let viewIdx = 0;

  const renderChips = () => {
    const { views } = curViews;
    if (views.length <= 1) {
      const ev = events[idx];
      const steps = (ev.v2 && Array.isArray(ev.v2.steps)) ? ev.v2.steps : [];
      const msg = steps.length ? `${steps.length} action(s): ${steps.map(s => s.action).join(", ")} (per-action boards not recorded in this replay)`
        : (String(ev.kind || "") === "turn" ? "no actions this turn" : "");
      chipsEl.innerHTML = `<span class="chips-empty" title="${esc(msg)}">${esc(msg)}</span>`;
      boardLabel.textContent = "";
      return;
    }
    chipsEl.innerHTML = views.map((v, i) =>
      `<span class="vchip ${v.cls || ""}${i === viewIdx ? " sel" : ""}" data-i="${i}" title="${esc(v.title || "")}">${esc(v.label)}</span>`).join("");
    const sel = chipsEl.querySelector(".vchip.sel");
    if (sel) {
      const l = sel.offsetLeft, r = l + sel.offsetWidth;
      if (r > chipsEl.scrollLeft + chipsEl.clientWidth) chipsEl.scrollLeft = r - chipsEl.clientWidth + 6;
      if (l < chipsEl.scrollLeft) chipsEl.scrollLeft = Math.max(0, l - 6);
    }
    const v = views[viewIdx];
    boardLabel.textContent = v ? (v.step ? stepTitle(v.step) : v.label) : "";
  };

  chipsEl.addEventListener("click", (e) => {
    const t = e.target.closest(".vchip");
    if (!t) return;
    viewIdx = Number(t.dataset.i) || 0;
    drawBoard(events[idx], curViews.views[viewIdx]);
    renderChips();
  });
  chipsEl.addEventListener("wheel", (e) => {
    if (Math.abs(e.deltaY) > Math.abs(e.deltaX)) { chipsEl.scrollLeft += e.deltaY; e.preventDefault(); }
  }, { passive: false });

  const renderBadges = (ev) => {
    const out = [];
    const notes = ev._notes || {};
    const nk = Object.keys(notes);
    if (nk.length) {
      const chars = nk.reduce((a, k) => a + k.length + String(notes[k]).length + 4, 0);
      out.push(`<span class="fn-tag fn-info">notes ${nk.length}<div class="sym-tip">${esc(`${nk.length} note(s), ${chars} chars\n\n` + nk.map(k => `${k}: ${notes[k]}`).join("\n\n"))}</div></span>`);
    }
    const store = ev._store;
    if (store && typeof store === "object" && Object.keys(store).length) {
      const ks = Object.keys(store);
      const tip = ks.map(k => { const s = store[k] || {}; return `${k}  [${s.type || "?"}${s.size != null ? " " + JSON.stringify(s.size) : ""}]\n  ${s.preview || ""}`; }).join("\n");
      out.push(`<span class="fn-tag fn-info">store ${ks.length}<div class="sym-tip">${esc("store (kept across run calls)\n\n" + tip)}</div></span>`);
    }
    if (ev.has_board_to_symbolic) {
      out.push(`<span class="fn-tag fn-yes">B→S${ev.symbolic_state_json ? `<div class="sym-tip">${esc(ev.symbolic_state_json)}</div>` : ""}</span>`);
    }
    if (ev.has_next_state !== undefined) out.push(`<span class="fn-tag ${ev.has_next_state ? "fn-yes" : "fn-no"}" title="next_state defined in the code file">next_state</span>`);
    if (ev.has_interpret_board !== undefined) out.push(`<span class="fn-tag ${ev.has_interpret_board ? "fn-yes" : "fn-no"}" title="interpret_board defined in the code file (does not prove it was called)">interpret_board</span>`);
    const ipStatus = ev.v2 && ev.v2.interpretation && ev.v2.interpretation.snapshot;
    if (ipStatus) out.push(`<span class="fn-tag ${ipStatus.status === "valid" ? "fn-yes" : "fn-no"}">interpretation: ${esc(ipStatus.status)}</span>`);
    badgesEl.innerHTML = out.join("");
  };

  // ── Simple markdown ──

  const simpleMd = (text) => {
    let h = esc(text);
    h = h.replace(/\*\*\*(.+?)\*\*\*/g, '<b><i>$1</i></b>');
    h = h.replace(/\*\*(.+?)\*\*/g, '<b>$1</b>');
    h = h.replace(/(^|[\s(])\*([^\s*][^*]*?[^\s*])\*([\s).,;:!?]|$)/gm, '$1<i>$2</i>$3');
    h = h.replace(/`([^`\n]+?)`/g,
      '<code style="background:rgba(255,255,255,0.06);padding:1px 4px;border-radius:3px;font-size:0.92em">$1</code>');
    h = h.replace(/^####\s+(.+)$/gm, '<b style="color:var(--text);font-size:12px">$1</b>');
    h = h.replace(/^###\s+(.+)$/gm, '<b style="color:var(--text);font-size:13px">$1</b>');
    h = h.replace(/^##\s+(.+)$/gm, '<b style="color:var(--text);font-size:14px">$1</b>');
    h = h.replace(/^#\s+(.+)$/gm, '<b style="color:var(--text);font-size:15px">$1</b>');
    h = h.replace(/^(\s*)[-•]\s+/gm, '$1<span style="color:var(--turquoise)">•</span> ');
    h = h.replace(/^(\s*)(\d+)\.\s+/gm, '$1<span style="color:var(--turquoise)">$2.</span> ');
    return h;
  };

  // ── Messages ──

  const renderHtmlBody = (el, html) => {
    if (html && html.trim()) { el.innerHTML = html; el.classList.remove("msg-empty"); }
    else { el.textContent = "(empty)"; el.classList.add("msg-empty"); }
  };

  const setMsgBody = (el, text) => {
    if (text && text.trim()) { el.innerHTML = simpleMd(text); el.classList.remove("msg-empty"); }
    else { el.textContent = "(empty)"; el.classList.add("msg-empty"); }
  };

  const imagesHtml = (ev) => {
    let imgBlock = "";
    if (Array.isArray(ev.images)) {
      for (const img of ev.images) {
        if (img && img.data_url) {
          const isBig = /\bbig\b/i.test(img.label || "");
          const cls = isBig ? "replay-img replay-img-big" : "replay-img replay-img-small";
          imgBlock += `<img class="${cls}" src="${img.data_url}" title="${esc(img.label || "Board image")}" alt="${esc(img.label || "Board")}"/>\n`;
        }
      }
    }
    return imgBlock;
  };

  const buildHarnessHtml = (ev, harnessText) => {
    let html = simpleMd(harnessText || "");
    if (ev.decision_policy) {
      const dp = ev.decision_policy;
      const uid = "dp_" + Math.random().toString(36).slice(2, 8);
      const forced = Array.isArray(ev.controller_actions) ? ev.controller_actions.join(", ") : "";
      html = `<div class="in-sec">Decision recovery: ${esc(dp.mode || "")} ${forced ? "— harness action: " + esc(forced) : ""}</div>`
        + `<div class="collapsible-hdr" onclick="this.classList.toggle('open');document.getElementById('${uid}').classList.toggle('open')">Recovery events / budgets</div>`
        + `<div class="collapsible-body" id="${uid}"><pre>${esc(JSON.stringify({request_options: ev.request_options, ...dp}, null, 2))}</pre></div>` + html;
    }

    // Inline images (the images sent with this turn's request)
    const imgBlock = imagesHtml(ev);
    if (imgBlock) html = imgBlock + html;

    // Tool definitions: collapsible at end of system prompt, with per-tool nesting
    if (ev.kind === "initial" || Array.isArray(ev.tool_definitions)) {
      const toolsMatch = html.match(/(\n?--- TOOL DEFINITIONS ---\n)([\s\S]*)/);
      if (toolsMatch) html = html.slice(0, toolsMatch.index);

      const outerUid = "td_" + Math.random().toString(36).slice(2, 8);
      if (Array.isArray(ev.tool_definitions) && ev.tool_definitions.length > 0) {
        let innerHtml = "";
        for (const tool of ev.tool_definitions) {
          const tid = "t_" + Math.random().toString(36).slice(2, 8);
          const name = esc(tool.name || "unknown");
          const desc = esc(tool.description || "");
          const json = esc(JSON.stringify(tool, null, 2));
          innerHtml += `<div class="collapsible-hdr" onclick="this.classList.toggle('open');document.getElementById('${tid}').classList.toggle('open')" style="margin:3px 0 1px">${name} <span style="color:var(--muted);font-weight:400;text-transform:none;font-size:9px;margin-left:6px">${desc}</span></div>` +
            `<div class="collapsible-body" id="${tid}"><pre style="margin:0;white-space:pre-wrap;word-break:break-word;font-size:10px;line-height:1.4">${json}</pre></div>`;
        }
        html += `\n<div class="collapsible-hdr" onclick="this.classList.toggle('open');document.getElementById('${outerUid}').classList.toggle('open')">Tool Definitions (${ev.tool_definitions.length} tools)</div>` +
          `<div class="collapsible-body" id="${outerUid}" style="padding:4px 6px">${innerHtml}</div>`;
      } else if (toolsMatch) {
        html += `\n<div class="collapsible-hdr" onclick="this.classList.toggle('open');document.getElementById('${outerUid}').classList.toggle('open')">Tool Definitions</div>` +
          `<div class="collapsible-body" id="${outerUid}"><pre style="margin:0;white-space:pre-wrap;word-break:break-word;font-size:10px;line-height:1.4">${toolsMatch[2]}</pre></div>`;
      }
    }

    return html;
  };

  // What the model was given for this event's request.
  const buildInputHtml = (ev, harnessText) => {
    const kind = String(ev.kind || "");
    if (kind === "initial") return buildHarnessHtml(ev, harnessText);
    if (kind === "game_over_reset" || kind === "level_reset" || kind === "compaction") {
      return `<div class="in-sec">Sent to the model <span>(${kind === "compaction" ? "pinned at the top of the context from now on" : "with the results of the turn before this event"})</span></div>`
        + simpleMd(kind === "compaction" ? String(ev.pinned_text || harnessText || "") : harnessText);
    }
    if (ev.request_text === undefined && ev.input_tool_text === undefined) {
      return `<span class="msg-empty">Not recorded in this replay (older harness version). The model's input for this turn was the previous turn's Harness Response, plus the images shown in Result.</span>`;
    }
    let h = "";
    const c = ev.context;
    if (c) {
      const turns = turnsText(c.first_turn, c.last_turn);
      h += `<div class="in-sec">Context <span>(besides the system prompt and tools)</span></div>`
        + `<div class="in-ctx">${c.pinned ? "pinned level-start message (see the last context-trim event) · " : ""}${esc(turns)}`
        + ` · ${esc(c.messages)} message(s) · ${esc(c.trims)} trim(s) so far</div>`;
    }
    if (ev.input_tool_text) {
      h += `<div class="in-sec">Tool result <span>(returned for the previous turn's tool calls)</span></div>` + simpleMd(String(ev.input_tool_text));
    }
    h += `<div class="in-sec">User message <span>(images, then text)</span></div>` + imagesHtml(ev) + simpleMd(String(ev.request_text || ""));
    return h;
  };

  const setPlainBody = (el, text) => {
    if (text && text.trim()) { el.textContent = text; el.classList.remove("msg-empty"); }
    else { el.textContent = "(empty)"; el.classList.add("msg-empty"); }
  };

  const OP_LABEL = { added: "+ added", updated: "~ updated", deleted: "− deleted", rejected: "✕ not saved" };

  const pinnedHtml = (ev) => {
    const notes = ev._notes || {};
    const ops = Array.isArray(ev._noteOps) ? ev._noteOps : [];
    const opByKey = {};
    for (const o of ops) opByKey[o.key] = o;
    const keys = Object.keys(notes);
    let h = "";
    if (keys.length || ops.length) {
      const chars = keys.reduce((a, k) => a + k.length + String(notes[k]).length + 4, 0);
      h += `<div class="pin-sec"><div class="pin-hdr">Pinned notes <span>· ${keys.length} entr${keys.length === 1 ? "y" : "ies"} · ${chars} chars${ops.length ? ` · ${ops.length} change(s) this turn` : ""}</span></div>`;
      for (const k of keys) {
        const o = opByKey[k];
        const badge = o && o.op !== "rejected" ? `<span class="note-op op-${o.op}">${OP_LABEL[o.op] || o.op}</span>` : "";
        h += `<div class="note-row${o ? " changed" : ""}">${badge}<span class="note-key">${esc(k)}</span>: ${esc(notes[k])}</div>`;
      }
      for (const o of ops) {
        if (o.op === "deleted") h += `<div class="note-row deleted"><span class="note-op op-deleted">${OP_LABEL.deleted}</span><span class="note-key">${esc(o.key)}</span>: ${esc(o.old || "")}</div>`;
        if (o.op === "rejected") h += `<div class="note-row"><span class="note-op op-rejected">${OP_LABEL.rejected}</span><span class="note-key">${esc(o.key)}</span>: ${esc(o.text || "")}</div>`;
      }
      h += `</div>`;
    }
    const log = Array.isArray(ev._eventLog) ? ev._eventLog : [];
    if (log.length) {
      const rowCls = (t) => /GAME OVER/.test(t) ? "go" : /RESET/.test(t) ? "rs" : /COMPLETED/.test(t) ? "ok" : "";
      const row = (t) => `<div class="elog-row ${rowCls(t)}">• ${esc(t)}</div>`;
      const tail = log.slice(-8);
      h += `<div class="pin-sec"><div class="pin-hdr">Event log <span>· written by the harness, pinned with the notes · ${log.length} entr${log.length === 1 ? "y" : "ies"}</span></div>`;
      if (log.length > tail.length) {
        const uid = "el_" + Math.random().toString(36).slice(2, 8);
        h += `<div class="collapsible-hdr" onclick="this.classList.toggle('open');document.getElementById('${uid}').classList.toggle('open')">${log.length - tail.length} earlier entr${log.length - tail.length === 1 ? "y" : "ies"}</div>`
          + `<div class="collapsible-body" id="${uid}">${log.slice(0, -8).map(row).join("")}</div>`;
      }
      h += tail.map(row).join("") + `</div>`;
    }
    return h;
  };

  let harnessMode = "result";
  try { harnessMode = localStorage.getItem("arcReplayHarnessMode") || "result"; } catch (e) { /* ignore */ }

  const renderMessages = (ev) => {
    const { summary, reasoning, llmOutput, harness } = parseMessages(ev);
    // Summary (this turn's visible text) + the pinned notes / event log as they stood after this event
    const pin = pinnedHtml(ev);
    if ((summary && summary.trim()) || pin) {
      summaryBody.innerHTML = (summary && summary.trim() ? simpleMd(summary) : `<span class="msg-empty">(no summary text)</span>`) + pin;
      summaryBody.classList.remove("msg-empty");
    } else {
      summaryBody.textContent = "(empty)"; summaryBody.classList.add("msg-empty");
    }
    setMsgBody(reasoningBody, reasoning);
    setPlainBody(llmBody, llmOutput);
    harnessSeg.querySelectorAll("button").forEach(b => b.classList.toggle("on", b.dataset.mode === harnessMode));
    harnessTitle.textContent = harnessMode === "input" ? "Model Input" : "Harness Response";
    renderHtmlBody(harnessBody, harnessMode === "input" ? buildInputHtml(ev, harness) : buildHarnessHtml(ev, harness));
  };

  harnessSeg.addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    harnessMode = b.dataset.mode;
    try { localStorage.setItem("arcReplayHarnessMode", harnessMode); } catch (err) { /* ignore */ }
    renderMessages(events[idx]);
  });

  // ── Code ──

  const renderCode = (ev, prevEv) => {
    const currCode = String(ev.latest_code || "");
    if (!currCode.trim()) {
      codeBody.innerHTML = `<div class="code-empty">No code written yet.</div>`;
      codeBadge.textContent = ""; codeBadge.className = "code-badge";
      return;
    }
    const prevCode = ev.previous_code !== undefined && ev.previous_code !== null
      ? String(ev.previous_code || "")
      : (prevEv ? String(prevEv.latest_code || "") : "");
    const changed = ev.code_changed !== undefined ? !!ev.code_changed : (currCode !== prevCode && prevCode !== "");

    if (changed) {
      codeBadge.textContent = "MODIFIED"; codeBadge.className = "code-badge modified";
      const diff = lineDiff(prevCode, currCode);
      let lineNum = 0;
      codeBody.innerHTML = diff.map(d => {
        let cls = "";
        if (d.type === "add") { cls = " line-add"; lineNum++; }
        else if (d.type === "del") { cls = " line-del"; }
        else { lineNum++; }
        const numText = d.type === "del" ? "−" : `${lineNum}`;
        return `<div class="code-line${cls}"><span class="ln">${numText}</span><span class="lc">${highlightPy(d.text)}</span></div>`;
      }).join("");
    } else {
      codeBadge.textContent = ""; codeBadge.className = "code-badge";
      const lines = currCode.split("\n");
      codeBody.innerHTML = lines.map((line, i) =>
        `<div class="code-line"><span class="ln">${i + 1}</span><span class="lc">${highlightPy(line)}</span></div>`
      ).join("");
    }
  };

  // ── Navigation ──

  let idx = events.length - 1;
  $("entityNames").addEventListener("change", () => drawBoard(events[idx], curViews.views[viewIdx]));

  const render = () => {
    const ev = events[idx];
    const prevEv = idx > 0 ? events[idx - 1] : null;
    curViews = buildViews(ev);
    viewIdx = Math.min(viewIdx, Math.max(0, curViews.views.length - 1));
    drawBoard(ev, curViews.views[viewIdx]);
    renderBadges(ev);
    renderChips();
    renderInfoBar(ev);
    renderMessages(ev);
    renderCode(ev, prevEv);
    slider.value = String(idx);
    turnText.textContent = `${turnLabelOf(ev, idx)}  ·  ${idx + 1} / ${events.length}`;
    prevBtn.disabled = idx <= 0;
    nextBtn.disabled = idx >= events.length - 1;
  };

  const goTo = (next) => {
    const clamped = Math.max(0, Math.min(events.length - 1, Number(next)));
    if (clamped === idx) return;
    idx = clamped;
    viewIdx = buildViews(events[idx]).def;
    render();
  };

  const stepView = (d) => {
    const n = curViews.views.length;
    if (n <= 1) return false;
    const next = Math.max(0, Math.min(n - 1, viewIdx + d));
    if (next !== viewIdx) {
      viewIdx = next;
      drawBoard(events[idx], curViews.views[viewIdx]);
      renderChips();
    }
    return true;
  };

  prevBtn.addEventListener("click", () => goTo(idx - 1));
  nextBtn.addEventListener("click", () => goTo(idx + 1));
  slider.addEventListener("input", () => goTo(Number(slider.value)));
  window.addEventListener("keydown", (e) => {
    if (e.target && (e.target.tagName === "INPUT" && e.target.type !== "range")) return;
    if (e.key === "ArrowLeft") { goTo(idx - 1); e.preventDefault(); }
    else if (e.key === "ArrowRight") { goTo(idx + 1); e.preventDefault(); }
    else if (e.key === "ArrowUp") { if (stepView(-1)) e.preventDefault(); }       // previous action board
    else if (e.key === "ArrowDown") { if (stepView(1)) e.preventDefault(); }      // next action board
    else if (e.key === "Home") goTo(0);
    else if (e.key === "End") goTo(events.length - 1);
  });

  let resizeTimer;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => { placeMarkers(); render(); }, 100);
  });

  viewIdx = buildViews(events[idx]).def;
  placeMarkers();
  render();
})();
