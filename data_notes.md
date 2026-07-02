# Data Notes: Startup Failure Dataset

## Files

| File | Rows | Cols | Sector covered | Has failure-reason flags? |
|---|---|---|---|---|
| `Startup Failures.csv` | 815 | 3 | all 13 sectors | no |
| `Startup Failure (Finance and Insurance).csv` | 47 | 20 | Finance and Insurance | yes (13 flags) |
| `Startup Failure (Food and services).csv` | 26 | 17 | Accommodation and Food Services | yes (10 flags) |
| `Startup Failure (Health Care).csv` | 60 | 20 | Health Care | yes (13 flags) |
| `Startup Failure (Manufactures).csv` | 30 | 20 | Manufacturing | yes (13 flags) |
| `Startup Failure (Retail Trade).csv` | 90 | 20 | Retail Trade | yes (13 flags) |
| `Startup Failures (Information Sector).csv` | 156 | 20 | Information | yes (13 flags) |

`Startup Failures.csv` is **not** a superset of the six enriched files — see "Master file is not a lookup table" below.

## Common columns

- `Name` (str) — company name
- `Sector` (str) — one fixed value per file (matches one of the 13 values in the master file's `Sector` column)
- `Years of Operation` (str) — see parsing notes below
- `What They Did`, `Why They Failed`, `Takeaway` (str, free text) — enriched files only
- `How Much They Raised` (str) — enriched files only, needs cleaning (see below)
- 10–13 binary (0/1) failure-reason flag columns — enriched files only

## Types

Everything loads as `str`/`object` except the binary flag columns, which pandas infers as `int64` (or `float64` where a null is present — see Quirk 5). Nothing is numeric out of the box: durations, funding amounts, and years all need parsing before they can be used in a regression.

## Quirks

**1. Two incompatible `Years of Operation` formats.**
- Finance, Health Care, Manufacturing files: `"YYYY-YYYY"` (no duration prefix)
- Food, Retail, Information, and the master file: `"N (YYYY-YYYY)"` (duration prefix included)
- Where both a prefix and a range are present, the prefix always equals `end - start` (0 mismatches found) — so it's safe to just always compute duration as `end - start` and ignore the prefix, giving one consistent numeric column across all files.

**2. `How Much They Raised` is inconsistent and sometimes not a real observation.**
Roughly 5–17% of rows per file (up to 27/156 in Information) don't match a clean `$<number><M|B|K>` pattern:
- Suffix annotations: `"$10M (est.)"`, `"$3M (est.)"` — estimated, not reported
- Attribution/context in parens, sometimes for the *wrong* entity: `"$1.7B (Dropbox)"`, `"$1.5M (Twitter $645M)"`, `"$0 (Yahoo $3.6B)"` — these mix the acquirer's or a comparator's number into the same cell as the target's raise
- `"$0 (Coinbase-funded)"` / `"$0 (Square-funded)"` — internally funded, not "$0 raised" in the normal sense
- Compound amounts: `"$15M+$34M pre-orders"`, `"$10M+$403M acquisition"` — two different quantities concatenated in one cell
- Free text instead of a number: `"$lowM"` (Health Care)
- All of these need a decision rule before parsing to numeric (e.g., regex-extract the first `$<num><unit>`, then manually review/drop the ambiguous "(Company)" and compound ones — they're a meaningfully different quantity, not noise).

**3. Failure-reason flag columns differ across files — not a fixed schema.**
- Food/services file has 10 flags and uses `High Operational Costs` instead of `Platform Dependency`/`Toxicity/Trust Issues`/`Regulatory Pressure`/`Overhype`, which the other five files have.
- The other five enriched files share the same 13 flag names.
- Any pooled regression across sectors (e.g., idea #2 from the earlier brainstorm) needs the Food file either excluded or reconciled (missing flags treated as structurally 0/NA, which is not the same as "confirmed absent").

**4. One missing value.** `Overhype` in the Finance file has 1 null, which silently upcasts that whole column to `float64`. All other flag columns are clean 0/1 with no rows summing to zero (every company has at least one reason flagged).

**5. Master file is not a lookup table for the enriched files.** Matching by exact `Name`:
- Finance: 0/47 missing from master
- Food: 1/26 missing
- Retail: 1/90 missing
- Information: 1/156 missing
- **Health Care: 53/60 missing**, and master only lists 32 Health Care companies total vs. 60 in the enriched file
- **Manufacturing: 21/30 missing**, master lists 46 Manufacturing companies vs. 30 in the enriched file

So for Health Care and Manufacturing especially, the two files look like independently compiled lists that happen to share a sector label, not a master/detail relationship. Don't assume you can join on `Name` to backfill sector-level context, and don't assume the master file's per-sector counts describe the enriched files' coverage.

**6. Duplicate row in the master file.** `Ouya` (Manufacturing, `4 (2012-2016)`) appears twice, identically — a straight duplicate, safe to drop one copy.

**7. Sector naming mismatch with common usage.** `Manufacturing` sector file is named `Startup Failure (Manufactures).csv` (typo/variant spelling) — doesn't affect data, only file discovery via glob patterns.

## Practical implications for analysis

- Build `duration_years` once as `end_year - start_year` from the parsed range; ignore the printed prefix.
- Build `raised_usd` by extracting the leading `$<num><unit>` via regex and converting M/B/K to a common unit (e.g. millions); treat the "(Company)" and compound-amount rows as needing manual review, not automatic parsing — they're likely a different quantity than what the column name promises.
- If pooling across sectors for the flags regression, either drop the Food/services file or explicitly document its 4 missing flag columns as structural NAs.
- Don't use `Startup Failures.csv` to enrich the sector files with sector/duration — each enriched file's own `Sector` and `Years of Operation` columns are already sufficient and are the reliable source for those rows.
