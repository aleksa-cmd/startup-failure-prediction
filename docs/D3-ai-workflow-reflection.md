# D3: AI-Workflow Reflection

## Which MCP servers / Claude Code features I used

**MCP servers:**
- **GitHub MCP** — attempted for repo creation. Its auth was broken in this environment (`mcp__github__create_repository` returned an auth error), so I fell back to the `gh` CLI, which was already authenticated. Verified the substitution produced an identical result (public repo, correct name, full commit history preserved) before moving on.
- **Vercel MCP** — used for deployment status checks (`get_deployment`) and a preview-access helper. Its `deploy_to_vercel` tool turned out to only print instructions rather than actually deploying, so the real deploy went through the `vercel` CLI instead.
- **Playwright MCP** — the workhorse of this project. Every UI feature was driven through a real browser, not just built and assumed to work: filling the predictor form, toggling flags, switching views, resizing to mobile widths, reading console logs for errors, and screenshotting for visual comparison against the design reference.

**Claude Code features:**
- **Skills** (`brainstorming`, `writing-plans`, `subagent-driven-development`) — used for every non-trivial build: clarify requirements and get explicit sign-off before writing code, turn the agreed design into an exact, code-complete plan, then execute it task-by-task.
- **Subagents** — each task in a plan was built by a fresh implementer subagent and checked by a separate fresh reviewer subagent with no memory of the implementer's reasoning, so the review couldn't just rubber-stamp the report.
- **Background processes** — local dev servers (Vite, `uvicorn`) run in the background so they can be driven live by Playwright without blocking the rest of the session.
- **AskUserQuestion** — used at real forks (static vs. live backend, which classifier to ship, GitHub-repo-now-or-later) rather than guessing.

## How I verified the AI's output

The core discipline: **reviewer subagents were told explicitly not to trust the implementer's self-report**, and to re-derive claims independently — reading source CSVs and recomputing numbers by hand, tracing render logic line-by-line instead of accepting "it works," and (for anything UI-facing) actually loading the page in a browser rather than reading the JSX and assuming it's correct.

That discipline caught real bugs that code review alone would have missed:
- A stale-state crash in the predictor when switching between regression and classification mid-session — found only because a subagent actually clicked through the flow in a browser.
- A missing `import './App.css'` that silently made an entire task's styling dead code for three review cycles, until a final side-by-side screenshot comparison against the design reference caught it.
- A `pyarrow` dependency the deployed model needed but nothing imported directly — found once by a live deploy failure, and again independently when a from-scratch clean-install test reproduced the same failure locally.

Numeric claims (which model actually won, what a feature's real importance is) were checked against the underlying data directly, not accepted from a report — including confirming that the design mockup's illustrative numbers didn't match the real computed ones, and using the real ones.

## Rough cost / effort

Three implementation plans went through this full cycle (feature pipeline + model ladders, the React/FastAPI app, and the visual redesign), each broken into 4–7 tasks. Each task cost two subagent dispatches at minimum (implementer + reviewer), more when a review found something to fix. Altogether this was on the order of **35–40 subagent dispatches** across the project. Cheap/fast models handled tasks where the plan already specified exact code (transcription); more capable models were reserved for anything needing real judgment — debugging a live failure, reviewing for correctness, or the final whole-branch reviews — matching effort to task difficulty rather than using the most expensive model everywhere.
