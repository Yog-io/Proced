# Agent Orchestration Brief: Build & Verify the Final Zenodo Dataset
### Self-contained — do not assume any prior conversation context beyond this document and the repo itself.

---

## 0. Who you are and what you're doing tonight

You are operating the `Proced` SAR oil-spill dataset pipeline (Block 1 of a maritime oil-spill detection & vessel-attribution project — SIH 2026). The team has already:
- Downloaded and fully extracted the Zenodo SAR dataset (multi-part `.7z` archives) to a local folder.
- Built and tested the pipeline (`proced/`, 68 tests passing) per `BLOCK1_DATA_TEAM_GUIDE.md` and `DATASET_FORMAT_REFINEMENT_REPORT.md`.
- Built and tested an independent QA suite (`qa_verification/`, 22 tests passing, 90/90 total) per `DATASET_VERIFICATION_ARCHITECTURE.md`, including a newly-added `raw_audit/` module for pre-pipeline raw-folder checks.

**Current scope is Zenodo only.** Do not process any Kaggle/OSD data in this run — that's a separate, later phase the team hasn't started.

**Your job tonight:** run the actual build — raw audit → full pipeline → handoff scripts → independent QA verification — against the real extracted dataset, and produce a clear status report the team can act on tomorrow morning. The human who owns this project is unavailable until tomorrow, so you need to make sound judgment calls about what to do autonomously and what to stop and flag — Section 4 below is the most important part of this document for that reason. Read it fully before starting.

---

## 1. Exact Execution Order

Do not skip steps or reorder them. Each step gates the next — don't proceed past a failed gate without following the decision rules in Section 4.

### Step 0 — Raw folder audit (before anything else)
```bash
python qa_verification/raw_audit/run_raw_folder_audit.py --root <path/to/extracted/zenodo/folder>
```
This is fast and standalone — it doesn't need any pipeline state. Read `raw_folder_audit_report.json` fully. **Do not proceed to Step 1 until you've read Section 4.1 below and applied its decision rules to whatever this reports.**

### Step 1 — Full pipeline run
The archive is already extracted, so skip the `extract` subcommand entirely — point `run` directly at the already-extracted tree:
```bash
python sar_dataset_pipeline.py run \
    --stage_dir <path/to/extracted/zenodo/folder> \
    --dest_dir data/master_dataset_processed \
    --output_archive data/master_dataset_v1.7z \
    --workers 8 \
    --seed 42
```
Use `--seed 42` for a reproducible run — record whatever seed you actually use in your final report regardless. This executes `scan → geo → features → plan → convert → metadata → split → archive → validate` in order and **exits with code 2 if the pipeline's own `validate` stage fails.** Capture stdout/stderr in full to a log file; you'll need it for the report either way.

### Step 2 — Handoff scripts
Run all three, in this order, from the repo root:
```bash
python scripts/join_wind.py            # defaults to --source mock — this is expected, see Section 4.3
python scripts/generate_ais.py
python scripts/generate_model2_pairs.py --pairs 2500   # defaults to --backend simple — expected, see Section 4.3
```

### Step 3 — Independent QA verification (against the real output, not fixtures)
```bash
QA_STAGE_DIR=<path/to/extracted/zenodo/folder> \
QA_DEST_DIR=data/master_dataset_processed \
QA_STATE_DIR=data/state \
QA_FEATURES_DIR=data/features \
QA_SPLITS_DIR=data/splits \
QA_CORRIDORS=data/reference/shipping_corridors.geojson \
python qa_verification/run_all_verification.py --seed 42
```
This exits 0 (pass), 2 (hard-fail), or 1 (harness error — a bug in the QA suite itself, not the dataset). Read `verification_report.json` in full.

### Step 4 — Produce the final status report
Write `BUILD_STATUS_<date>.md` (see Section 5 for the exact template) summarizing everything, regardless of whether the run succeeded, partially succeeded, or had to stop early. **Always produce this report — never end the session without one, even if you had to halt partway through.**

---

## 2. What "done" means tonight

Only declare the dataset build fully complete if **all** of the following hold:
1. Step 0's raw audit report shows `ready for pipeline: yes` (or you applied Section 4.1's quarantine rule and documented exactly what was excluded and why).
2. Step 1's pipeline `run` exits 0 (not 2).
3. Step 3's `run_all_verification.py` exits 0 (zero hard-fails). Soft-fails are fine and expected — list them in the report, don't let them block completion.
4. The final report (Step 4) is written.

If any of 1–3 isn't true, that's fine — this is still a normal outcome for an overnight run. Stop per Section 4's rules, write the report honestly reflecting where things stand, and let the human pick it up in the morning. **A clear, honest "here's exactly where it stopped and why" report is a successful night's work. A forced "done" that glosses over an unresolved hard-fail is not.**

---

## 3. Guardrails — things you must never do, regardless of what seems expedient

- **Never edit `proced/` or `qa_verification/` source code to make a failing check pass**, unless the failure is unambiguously a trivial environment/path issue (e.g., a wrong default path in a command you're running, not a logic bug) — and even then, log exactly what you changed and why in the report. If a check fails because of an actual logic issue in the code, that's a finding to report, not something to silently patch.
- **Never delete, move, or overwrite the raw extracted source dataset.** Quarantining a corrupt file (Section 4.1) means copying its path into a report/manifest, not deleting the original.
- **Never mark verification as passed if `verification_report.json` shows any hard-fail**, even if you believe the hard-fail is a false positive — flag your reasoning in the report instead, but don't override the result.
- **Never fabricate or guess at missing real-world data** (AIS, wind, currents). The `join_wind.py --source mock` and `generate_model2_pairs.py --backend simple` defaults exist precisely so the pipeline can produce a complete, clearly-flagged placeholder rather than you inventing values — always use the documented mock/simple fallback, never hand-construct substitute data yourself.
- **Never retry a failed step more than once with an automatically-adjusted parameter.** One bounded retry is allowed only when Section 4 explicitly says so and the fix is unambiguous from the error message. Beyond that, stop and report — don't loop.
- **Never run `extract`** — the archive is already extracted; running it again risks duplicating or corrupting the existing extraction.

---

## 4. Decision Rules — how to handle issues without waiting for a human

### 4.1 Raw audit (Step 0) finds problems
- **A handful of corrupt/unopenable files (well under 5% of total scenes):** quarantine — record their paths in the report, exclude them from the run by noting them, and proceed to Step 1 on the remaining clean data. This is a normal, expected outcome; don't stop for it.
- **A large fraction of files corrupt or unopenable (over ~5%), or entire subfolders missing/suspiciously empty relative to siblings:** this suggests an incomplete multi-part `.7z` extraction. **Stop here. Do not proceed to Step 1.** This isn't something to work around — a systematically incomplete extraction needs the human to check whether a `.7z` part is genuinely missing or corrupted at the source. Write the report and wait.
- **Orphaned images/masks (pairing mismatches):** proceed — per the guide's own rule, a scene with no valid mask pairing is simply treated as unlabeled by the pipeline itself. Log the count in your report; this doesn't block anything.
- **Duplicate files detected:** log them, proceed — duplicates don't corrupt the pipeline's output (later stages will just process the same scene twice under two paths), but flag the count so the human can decide whether to dedupe the source folder later.

### 4.2 Pipeline `run` (Step 1) exits 2 (validation failure)
- Read `pipeline_summary.json`'s validation section for the *specific* reason.
- **If the reason clearly matches a known, already-documented issue category** (e.g., a value-domain misdetection you can see is wrong from the file's actual header info, fixable via `--calibrated-match`/`--uncalibrated-match`/`--value-domain` override flags that the pipeline already exposes) — you may retry **once** with the corrected flag, and must document in the report exactly what you observed and what override you applied.
- **If the reason is anything else, or your one retry also fails:** stop. Do not attempt a second workaround. Write the report with the exact error output included verbatim, and wait for the human.

### 4.3 Handoff scripts (Step 2)
- These are **expected to use mock/simple defaults** in this environment (no live CDS credentials, no real current/wind forcing files configured) — this is not a failure, it's the documented honest-limitation path. Confirm the output files exist and are clearly flagged as mock/synthetic (`is_synthetic_location`, `synthetic_*` field prefixes, etc.), then proceed. Do not attempt to source real credentials or real forcing data yourself.

### 4.4 QA verification (Step 3) reports hard-fails
- **Do not attempt to fix the underlying data or code.** Your job is to characterize the failure precisely, not resolve it.
- For each hard-fail category, capture: which specific check failed, how many rows/files it affected, and a representative example (file path + the specific mismatch found) — this is what turns a vague "something's wrong" into something the human can act on in five minutes tomorrow instead of fifty.
- If a hard-fail affects only a small number of specific rows (not a systemic pattern), you may exclude just those rows from a "provisionally usable subset" note in the report — but do not repackage/re-archive a "fixed" dataset yourself. That decision belongs to the human.
- **Soft-fails** (class imbalance, visual spot-check flags, etc.) never block anything — just list them clearly in the report.

### 4.5 Anything not covered above
If you hit a situation this document doesn't address — an error message you don't recognize, a check behaving unexpectedly, anything ambiguous — **do not guess and do not improvise a fix.** Stop what you're doing, write down exactly what you observed (full error text, what step you were on, what you'd tried), and move on to producing the final report. An honest "I don't know what caused this, here's everything I saw" is far more useful to the human tomorrow than a guessed workaround that might have silently changed something in the data.

---

## 5. Final Report Template — `BUILD_STATUS_<date>.md`

```markdown
# Dataset Build Status — <date>

## Summary verdict
[ ] Fully complete and verified (all Section 2 conditions met)
[ ] Stopped partway — see below for exactly where and why

## Step 0 — Raw Audit
- Total files discovered: 
- Corrupt/unopenable: 
- Orphaned pairs: 
- Duplicates: 
- Action taken: [proceeded / quarantined N files / STOPPED — incomplete extraction suspected]

## Step 1 — Pipeline Run
- Command used (exact, with all flags):
- Exit code:
- If exit 2: paste the validation failure reason verbatim
- Retry attempted? [no / yes — describe exact override and outcome]
- Final counts: oil / look-alike / background instances; train/val/test split sizes

## Step 2 — Handoff Scripts
- join_wind.py: [source used, mock or era5]
- generate_ais.py: [output confirmed present]
- generate_model2_pairs.py: [backend used, pair count]

## Step 3 — QA Verification
- Exit code:
- Hard-fails (must be empty for a complete build): list each, with affected count + example
- Soft-fails: list each

## Open items for the human tomorrow
[Anything from Section 4 you stopped on, or anything you're unsure about — be specific, include file paths and exact error text, not summaries]

## Full logs
[Attach or reference the raw stdout/stderr logs from each step]
```

---

## 6. One last reminder

Slow and honest beats fast and uncertain here — this dataset feeds directly into Model 1, the look-alike classifier, and Model 2's training, so a silently-wrong "success" tonight costs the whole team more time tomorrow than a clearly-documented stop would. When in doubt, stop and write it down rather than push forward.
