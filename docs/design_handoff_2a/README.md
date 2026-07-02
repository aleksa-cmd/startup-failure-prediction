# Handoff: Startup Failure Predictor — "Studio" Redesign (option 2a)

## Overview
A full visual redesign of the existing two-view app at `startup-failure-prediction.vercel.app` (React 19 + Vite, plain CSS, `recharts`, FastAPI backend). Two views:

1. **Predictor** — two-pane layout: form on the left, an always-visible **live prediction panel** on the right that updates as the user edits inputs (debounced API calls — no submit round-trip).
2. **Dashboard** — model-evaluation results, designed so the editorial point "the simple Decision Tree beat XGBoost" is visually unmissable.

The API contract is unchanged: `POST /api/predict/regression` and `POST /api/predict/classification` with the existing JSON shapes.

## About the Design Files
`2a-reference.html` in this bundle is a **design reference created in HTML** — a static prototype showing intended look, spacing, and copy. It is NOT production code. The task is to **recreate this design inside the existing React 19 + Vite codebase**, replacing the leftover Vite scaffold CSS, keeping `recharts` for the dashboard charts, and wiring the form to the existing FastAPI endpoints.

## Fidelity
**High-fidelity.** Colors, typography, spacing, radii, and copy are final. Recreate pixel-perfectly. Values below are exact.

## Design Tokens

Colors (define as CSS custom properties on `:root`):
- `--bg: #f6f5f3` — page background (warm off-white)
- `--surface: #ffffff` — cards
- `--ink: #17161a` — primary text / dark fills
- `--ink-soft: #44423c` — secondary text
- `--muted: #8b877e` — labels, captions
- `--faint: #b9b6ae` — de-emphasized text
- `--line: #e4e1db` — borders on unselected chips
- `--track: #f0eeea` — bar-chart track background
- `--bar-muted: #c9c5bd` — non-winner bars
- `--accent: oklch(0.55 0.2 275)` — primary purple (fallback `#5b4fd6`-ish; keep oklch, all modern browsers support it)
- `--accent-deep: oklch(0.42 0.18 275)` — text on white inside accent contexts
- `--accent-soft: oklch(0.55 0.2 275 / .1)` — selected-chip fill
- Divider inside accent panel: `rgba(255,255,255,.25)`

Typography:
- Single family: **Space Grotesk** (Google Fonts, weights 400/500/600/700). Load via `<link>` in `index.html`.
- Page H1: 38px / 700 / line-height 1.05 / letter-spacing -0.02em
- Big prediction number: 64px / 700 / letter-spacing -0.02em (unit suffix: 22px / 500 / 80% opacity)
- Section labels (eyebrows): 12px / 600 / uppercase / letter-spacing .08em / `--muted`
- Card titles: 14–15px / 700
- Body: 13–13.5px / line-height 1.5
- Captions: 11–12.5px

Spacing & shape:
- Cards: `border-radius: 18px`, `box-shadow: 0 2px 16px rgba(20,18,30,.06)`, padding ~28px 30px
- Small tiles/chips: `border-radius: 12px`
- Pills (nav, toggles): `border-radius: 999px`
- Grid gaps: 20px between panes, 8–12px inside grids
- Page content max-width ~980px, horizontal padding 40px

## Screens / Views

### Shared header (both views)
- Flex row, space-between, padding 20px 40px, on `--bg` (no card).
- Left: wordmark `startup/fail` — 16px/700, the `/fail` part in `--accent`.
- Right: two pill buttons, gap 8px: **Predict** and **Results** (13px). Active pill: `--ink` background, white text, weight 600. Inactive: transparent, `1.5px solid #d9d6d0` border, weight 500.
- These switch views (existing tab state / router).

### View 1: Predictor
- H1 below header: two lines — `Describe a startup.` in `--ink`, then `Watch the prediction move.` in `--accent`.
- Main grid: `grid-template-columns: 1.3fr 1fr; gap: 20px; align-items: stretch`.

**Left pane — form card** (white card, three stacked groups, 24px gap):

1. *What to predict* (eyebrow label). Two-column grid of selectable tiles, gap 8px:
   - Selected tile: `--accent-soft` fill, `1.5px solid --accent` border, title 13.5px/700 in `--accent-deep`, subtitle 11.5px `--muted`.
   - Unselected: `1.5px solid --line`, title in `--ink-soft`.
   - Tiles: "⏱ Years it survives / regression" and "Early vs. late failure / early / typical / long-run". Maps to the regression-vs-classification radio.
2. *The basics* (eyebrow). Grid `1fr 1.3fr 1fr`, gap 10px. Each field is a tile: `--bg` fill, radius 12px, padding 12px 14px, tiny uppercase 11px/600 `--muted` label on top, value 17px/700 below (sector shows value 15px/600 + `▾` glyph).
   - Raised → number input ($M). Sector → `<select>` with the 6 existing options (style the closed state as the tile; a natively-opened dropdown is fine). Founded → number input (year).
3. *What's going wrong?* — header row: eyebrow left, right-aligned live counter `N of 9 selected` (12px/700, `--accent-deep`).
   - 3×3 grid of toggle chips, gap 8px, radius 12px, padding 11px 13px.
   - Unselected: `1.5px solid --line`, title 12.5px/600 `--ink-soft`, subtitle 11px `--muted`.
   - Selected: solid `--ink` fill, white title, subtitle `#b9b6ae`. (Selected = dark, NOT purple — purple is reserved for the prediction panel.)
   - The 9 chips (title / subtitle):
     - 🏰 Giants / big incumbent
     - 💸 No Budget / out of runway
     - ⚔️ Competition / crowded market
     - 🎯 Poor Market Fit / nobody wants it
     - 🧊 Acq. Stagnation / went nowhere
     - 🪙 Monetization / can't charge
     - 🕳 Niche Limits / market too small
     - 🔧 Execution / flawed delivery
     - 🌊 Trend Shifts / market moved on
   - Chip → API mapping uses the existing full reason names (e.g. "Acquisition Stagnation", "Monetization Failure", "Execution Flaws").

**Right pane — live prediction panel** (fills column height, `display:flex; flex-direction:column`):
- Solid `--accent` background, white text, radius 18px, padding 28px.
- Eyebrow: `LIVE PREDICTION` (75% opacity).
- Regression result: huge number `7.2` (64px/700) + ` yrs` suffix; below it one sentence, 13.5px, 85% opacity, e.g. "A typical run. Most startups like this fold between years 5 and 10." (pick the sentence by predicted bucket: <5 early, 5–10 typical, >10 long-run).
- Divider: 1px `rgba(255,255,255,.25)`, 22px vertical margin.
- Eyebrow: `WHERE THAT FALLS`. Then a 36px-tall segmented band, radius 10px, overflow hidden, three segments sized 32% / 41% / 27%, 11px/600 labels: `EARLY <5y`, `TYPICAL 5–10y`, `LONG >10y`. Inactive segments: `rgba(255,255,255,.14)` fill, 70% opacity text. The active segment (where the prediction falls): solid white fill, `--accent-deep` text.
- Footnote 12.5px, 75% opacity: "Updates as you type — no submit step. Decision Tree, trained on 400 real startup post-mortems (test R² 0.38)."
- Bottom (pushed down with `margin-top:auto`): white button, radius 12px, padding 13px, centered, 14px/700 `--accent-deep`: `Re-run prediction →` (manual fallback trigger).

**Classification mode:** when "Early vs. late failure" is selected, the panel swaps the big number for the predicted class name (e.g. `typical`, 40–48px/700) and the segmented band becomes the probability readout — segment widths proportional to the 3 class probabilities, each labeled `EARLY 23%` etc., active = highest probability (white fill). Keep everything else identical.

### View 2: Dashboard
Same header (Results pill active). Content stacks with 12–14px gaps:

1. **Top stat row** — grid `1.4fr 1fr 1fr`, gap 12px:
   - Headline card: solid `--accent`, white; eyebrow `HEADLINE`; text 19px/700 line-height 1.3: "The simple Decision Tree beat XGBoost on regression 🏆".
   - Stat card: white; eyebrow `BEST R²`; number `0.38` 34px/700; caption "Decision Tree · years survived".
   - Stat card: white; eyebrow `BEST F1`; number `0.60` 34px/700; caption "XGBoost · timing (Tree: 0.57)".
2. **Two leaderboard cards** side by side (grid `1fr 1fr`):
   - Card title 14px/700 + caption 12.5px `--muted`.
   - "Round 1: predicting years survived" / "R² — guessing the average scores 0":
     rows (label 110px col, bar, value 38px col; 10px gaps): `Tree 🥇` .38 (accent bar, label `--accent-deep` 700) · `XGBoost` .37 (`--bar-muted`) · `Lasso` .36 (`--bar-muted`) · `Guessing` .00 (near-zero sliver, `--line`, label `--faint`).
   - "Round 2: timing the failure" / "Early / typical / long-run · F1 score":
     `XGBoost` .60 (bar in `--ink`, value 700) · `Tree 🥈` .57 (accent bar) · `LogReg` .55 (`--bar-muted`) · `Guessing` .19 (`--line`).
   - Bars: 13px tall, full-round (999px), on `--track` track. Widths proportional to value relative to the card's max (e.g. Round 1: 95/92/90/1.5%; Round 2: 100/95/91/31%).
   - Implement with `recharts` horizontal `BarChart`s (existing dependency) styled to match — rounded bars, no axes/gridlines, labels as the category axis — or as plain flex/div bars if simpler; visual result must match.
3. **Feature-importance card** (full width, white):
   - Header row: title "What actually matters" left; right, a two-pill toggle: `Years survived` (active: `--ink` fill, white) / `Failure timing` (inactive: border `--line`, `--muted` text). Toggling swaps the dataset (regression permutation importance vs. classification permutation importance).
   - Rows, grid `130px 1fr 80px`: `When founded` (100%, accent, annotated "strongest" in the right column, 12px `--muted`) · `Money raised` (54%, accent at 55% opacity) · `Sector` (11%, `--bar-muted`) · `Giants pressure` (6%) · `No Budget pressure` (4%). For the classification dataset use: Money raised (100%), When founded (~68%), Sector (~25%), big-tech pressure (~17%), No Budget (~9%) — rescale from the real permutation numbers.
   - Footer note, 12.5px `--muted`, above a 1px `--track` top border: "**Surprise:** the era a startup was born in matters more than any reason it failed." (bold part in `--ink`).
   - Human-readable feature labels only (no `decade_started` / `raised_musd` raw names).

## Interactions & Behavior
- **Live prediction:** on any form change, debounce ~400ms, then POST to the endpoint matching the selected prediction type; while in flight, dim the big number to 60% opacity (no spinner). On error, show "Couldn't reach the model — try again" at 13px in the panel and keep the last good result.
- Fire one prediction on initial load with the defaults (Raised 10, Finance and Insurance, 2015, no flags) so the panel is never empty.
- Chip/tile toggles: instant visual state change; cursor pointer; hover on unselected chips/tiles: border-color `#c9c5bd`.
- Prediction-type tiles switch the right panel between regression and classification layouts (see above).
- Buttons/pills hover: 90% opacity on solid fills.
- No page-level animations required; a 150ms ease transition on chip background/border and on bar widths is enough.
- Responsive: below ~900px stack the two predictor panes vertically (prediction panel first), stat row and leaderboards to single column. Chip grid 3→2 columns below 700px.

## State Management
- `predictionType: 'regression' | 'classification'`
- `raised: number`, `sector: string`, `founded: number`, `flags: Set<string>` (the 9 reason names)
- `result: { years?: number, cls?: string, probs?: {early,typical,long_run} } | null`, `loading`, `error`
- Dashboard: `featureView: 'regression' | 'classification'`; chart data is static (hardcoded from the eval results — no fetch needed).

## Assets
- Google Font: Space Grotesk (400–700).
- Icons are plain emoji characters in the chip labels — no icon library needed.
- No images.

## Files
- `2a-reference.html` — self-contained static reference of both views in this bundle. Open in a browser and match it.
