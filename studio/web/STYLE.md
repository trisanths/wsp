# Sapien visual style

The studio follows the look of a calm, document-like agent workspace: monochrome surfaces, a serif display headline, system UI text, hairline borders, a left sidebar, and a composer docked at the bottom. The values below were measured on a reference agent app (manus.im) on 2026-11-14 and are used here as a palette and type scale only. Sapien keeps its own name, wordmark, icons, illustrations, and copy. Do not copy any logo, product name, image asset, or text from the reference.

## Themes

Both themes ship. The default follows `prefers-color-scheme`; a toggle in the sidebar footer overrides it and is remembered in localStorage (wrapped in try/catch). Set the theme as a class on `<html>` (`light` or `dark`).

| Token | Light | Dark | Use |
|---|---|---|---|
| `--bg` | `#ffffff` | `#242424` | main area |
| `--bg-sidebar` | `#f8f8f7` | `#1f1f1f` | sidebar |
| `--bg-sunk` | `#f8f8f7` | `#1a1a1a` | wells, code, canvas |
| `--bg-card` | `#fafafa` | `#1c1c1c` | cards |
| `--bg-raised` | `#ffffff` | `#313131` | menus, popovers, the floating bar |
| `--text-1` | `#1a1a1a` | `#e6e6e6` | primary text |
| `--text-2` | `#4d4d4d` | `#b3b3b3` | secondary text |
| `--text-3` | `#737373` | `#8c8c8c` | tertiary text, icons |
| `--text-4` | `#b9b9b7` | `#595959` | disabled and decoration only, never text people need to read |
| `--line-1` | `#0000000a` | `#ffffff08` | faint dividers |
| `--line-2` | `#0000000f` | `#ffffff0f` | borders on cards and inputs |
| `--line-3` | `#00000024` | `#ffffff1f` | strong borders, the composer |
| `--line-focus` | `#0000004d` | `#ffffff52` | focused inputs |
| `--fill-1` | `#37352f05` | `#ffffff0a` | row hover |
| `--fill-2` | `#37352f0a` | `#ffffff0f` | selected row, chips |
| `--fill-3` | `#37352f14` | `#ffffff14` | pressed |
| `--btn-bg` | `#1a1a19` | `#fffffff2` | primary button |
| `--btn-fg` | `#ffffff` | `#1a1a1a` | primary button text |
| `--blue` | `#0081f2` | `#1a93fe` | running state, progress, links, the only accent |
| `--green` | `#3da546` | `#61aa64` | success, "beats LoRA" |
| `--red` | `#f25a5a` | `#e1474b` | errors, collapse, guard floors |
| `--amber` | `#efa201` | `#dca451` | LoRA bar, warnings |
| `--purple` | `#8269cb` | `#8269cb` | one chart series only |
| `--shadow-card` | `0 0 0 1px #0000000f, 0 1px 2px #00000008` | `0 0 0 1px #ffffff24, 0 1px 2px #00000047` | cards |
| `--shadow-float` | `0 12px 32px #00000014` | `0 12px 32px #00000066` | composer, floating bar, drawer |

No gradients on UI chrome. Colour lives in data (bars, lines, status dots) and in the small accent set above. The old purple gradient buttons go away: primary buttons use `--btn-bg` / `--btn-fg`.

## Type

- Text: `-apple-system, system-ui, "Segoe UI", Helvetica, Arial, sans-serif`.
- Display: `"Libre Baskerville", Georgia, Cambria, "Times New Roman", serif`, weight 400, from Google Fonts. Use it only for page headlines (home, spec title, run title, builder title) and large numbers in KPI tiles.
- Mono: `ui-monospace, "SF Mono", Menlo, monospace` for program text and ids.

| Role | Size / line | Weight |
|---|---|---|
| Home headline | 32px / 40px serif | 400 |
| Page title | 24px / 30px serif | 400 |
| Section title | 16px / 24px | 500 |
| Body, inputs | 16px / 24px | 400 |
| Rows, buttons, labels | 14px / 20px | 400 or 500 |
| Small, meta | 12px / 16px | 400 |

## Shapes and sizes

- Sidebar 300px wide, full height, `--bg-sidebar`, no border (the colour step separates it). Rows 36px tall, radius 10px, padding 0 8px, 14px text, an 18px stroke icon then the label with an 8px gap. Hover `--fill-1`, selected `--fill-2`. The sidebar collapses to a 56px icon rail with a toggle (the builder starts collapsed).
- Main top bar: 52px, no border, transparent. Left: a quiet picker-style label (for example "gpt-5.4-mini agents" with a chevron, opening a small read-only menu that shows backend and LLM). Right: compact pills such as evals left.
- Cards: radius 12px, `--bg-card`, `--shadow-card`, padding 16px 12px for small entry cards and 16px for panels.
- Composer: 768px max width, radius 22px, 1px `--line-3` border, `--bg` fill, `--shadow-float`, padding 12px 16px. Bottom row: round 32px icon buttons (attach) on the left, a round 32px send button on the right that is `--fill-2` when empty and `--btn-bg` when there is text.
- Buttons: primary 32 to 36px tall, radius 8px, 14px/500. Secondary: transparent with a 1px `--line-2` border. Icon buttons: 32px round.
- Chips and tags: radius 9999px, 1px `--line-2` border, 12 to 14px text. Status tags use the accent colour at 10 to 12% alpha fill with the solid colour as text.
- Inputs and sliders: thin 4px tracks in `--fill-3`, filled part `--text-1` (or the status colour for guards), 16px knob in `--bg` with a `--line-3` border and `--shadow-card`. Tick marks are 2px wide, labels 11px in `--text-3`.
- Icons: stroke icons, 18px in rows and 16px in buttons, stroke width 1.75, colour `--text-3` (or `--text-1` when active).
- Motion: 120 to 180ms ease for hovers and panels; no bouncing.

## Layout per screen

- App shell: sidebar on the left, main area on the right. Sidebar content: the Sapien wordmark (our own text "Sapien" with a simple geometric mark we draw), then rows "New model", "Runs", "Builder", "Parts library", then a "Recent" heading (12px, `--text-3`) and the last specs and runs the viewer opened (kept in localStorage), then a footer with the theme toggle and the backend status (STUB or Modal GPU).
- Home: headline centred in the upper third, a row of entry cards under it (each with a coloured 20px stroke icon and a 14px/500 label), example chips under the cards, and the composer docked near the bottom of the main area. Attachments show as chips inside the composer.
- Spec: a centred document column (max 720px) with the serif title, the summary, and sections with 16px/500 headings. The "how do you want to build it" choice sits in a sticky card at the bottom like the composer.
- Run: header with the run title in serif and status. Agent cards on the left of the main area, stats on the right, the floating progress bar docked at the bottom like the composer (radius 22, `--bg-raised`, `--shadow-float`).
- Agent drawer: a panel that slides in from the right over the run (about 50% width), styled like an agent's own screen: a title row with the agent name and status, then a dark well (`--bg-sunk`) with the program it is writing and its tool calls, then its thinking as a plain text list, then a nudge composer styled like the main composer.
- Builder: the sidebar starts as a rail. The palette is a second column with search and groups. The canvas uses `--bg-sunk` with a faint dot grid in `--line-2`. Host layer slabs are cards. The floating action bar is a composer-style pill. The right panel uses the card style.

## Run screen as a task view (measured 2026-11-14 on a live task)

The run screen follows the reference's task view: a conversation column on the left and a tabbed workspace panel on the right.

- Main area background in the run view is `--bg-sunk`. The conversation column is centred, about 560px wide when the panel is open and 736px when it is closed.
- Messages:
  - The spec (and every nudge the user sends to the whole run) shows as a user message: right-aligned bubble, `--bg` fill on the `--bg-sunk` background, radius 12px 12px 4px 12px (the bottom-right corner is 4px), padding 12px, 16px/24px text, max width about 480px.
  - Agent updates (analyst maps, planner hypotheses, judge verdicts, scored evals, collapses) show as agent messages: a header row with the agent's 24px avatar, its name in 16px/500, and a role tag (12px, `--text-3`, 1px `--line-3` border, radius 6px, padding 2px 6px), then the text in 16px/24px with no bubble. Scored evals show a compact score line with coloured status. Group consecutive messages from one agent under one header.
  - A thinking state shows "<agent> is thinking" in `--text-3` with three animated dots.
- The composer docks at the bottom of the conversation column (same style as home). Sending text there nudges the whole run (all planners); a small agent picker on the composer's left can target one agent. Under it, a 12px `--text-3` line: "Agents can be wrong. Check results on the held-out split before you use them."
- Workspace panel on the right, about 600px wide, separated by a 1px `--line-2` divider, with an expand button (full width) and a close button in its top-right corner. A toggle button in the main top bar reopens it.
  - Top row is a tab strip: 32px chips, radius 8px, 13px text, `--fill-2` for the active tab, transparent for others, with a small close x on agent tabs. The first tab is "Overview" (fixed). Clicking an agent card or an agent's name in the conversation opens that agent as a tab.
  - Overview tab: the agent cards (as a compact grid or list), the KPI tiles, the leaderboard, the best-score chart, and the sources list.
  - Agent tab: the agent's own screen. Status row, then a dark well with the program it is writing and its tool calls, then its thinking, its history, and a nudge box for that agent only.
  - The progress bar docks at the bottom of the panel, styled like the composer (radius 22px, `--bg-raised`, `--shadow-float`), with Eval N of M, compute, best vs target, Pause/Resume and Stop and keep best.
- On narrow widths (under 1100px) the panel overlays the conversation instead of splitting.
