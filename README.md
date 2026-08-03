<p align="center">
  <img src="assets/banner.png" alt="Hermes Agent" width="100%">
</p>

# hermes-tabular ☤

A small, opinionated pack of **Hermes skills** for tabular intelligence.

Ten skills. One entry point. A deliberately narrow decision space so an agent can consistently make good choices instead of drowning in options.

---

## Why this exists

Existing tabular tools tend to fall into two extremes. Some are too narrow, forcing a single model backend onto every dataset. Others are too wide, drowning the agent in endless hyperparameter grids, redundant gradient boosting backends, and samplers. The agent spends its token budget picking knobs instead of solving the problem.

`hermes-tabular` sits in the **goldilocks zone**: a deliberately structured decision space built on four distinct model families. It gives the agent room to adapt to your data without wasting budget on unnecessary choices:

| Family | Default | Why |
| --- | --- | --- |
| Statistical | Linear / Logistic Regression | Fast, interpretable baseline, calibration reference |
| Tree-based ML | CatBoost | Strong default for most tabular data with mixed feature types |
| Foundation Model | TabICL | Excellent for small and medium datasets with minimal tuning |
| Ensemble | Simple averaging / stacking | Frequently the best final performer |

Additional backends can be added later as new skills without touching orchestration policy.

---

## Repository layout

```
hermes-tabular/
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
│
├── references/
│   └── pitfalls.md
│
├── examples/
│   ├── titanic.md
│   ├── california_housing.md
│   └── churn.md
│
├── scripts/                  # human-side convenience only; agents do not need these
│   ├── install_skills.py
│   └── validate_skills.py
│
└── skills/
    ├── tabular/              # orchestrator (the only skill invoked directly)
    │   ├── SKILL.md
    │   └── defaults.yaml
    ├── tabular-inspect/
    ├── tabular-load/
    ├── tabular-split/
    ├── tabular-baseline/
    ├── tabular-catboost/
    ├── tabular-tabicl/
    ├── tabular-ensemble/
    ├── tabular-evaluate/
    ├── tabular-uncertainty/
    └── tabular-checkpoint/
```

---

## Architecture Pattern

+-------------------------------------------------------------+
| System Environment (Human Administrator)                    |
| Handled once via pip/uv. Installs heavy binary backends:    |
| - CatBoost, PyTorch, TabICL, Conformal Prediction           |
+-------------------------------------------------------------+
                              │
                              ▼
+-------------------------------------------------------------+
| Skill Injection (Hermes Agent / Harness)                    |
| Handled by running scripts/install_skills.py:                |
| - Copies/symlinks markdown prompts, rules, & defaults      |
+-------------------------------------------------------------+
---

## Install

**Two different audiences do two different jobs here.** Read only your section.

| | Who | Does what | How often |
| --- | --- | --- | --- |
| **Humans** | operator / maintainer | provision the Python environment | once per machine |
| **Agents** | Hermes | place and remove skill files in a project | per project, freely |

The split exists because installing `torch` and CatBoost is a privileged, heavy,
network-dependent operation that an agent should not be improvising. Copying a
directory of Markdown files is not. Keep those concerns apart and neither audience
has to think about the other's job.

---

### For Humans / System Setup

One-time. Provision the runtime the skills assume, then get out of the way.

**1. Install Python dependencies.**

```bash
pip install -r requirements.txt
```

Or install only what you intend to use — the foundation model pulls `torch` and is
by far the heaviest piece:

```bash
pip install -e ".[catboost]"      # CatBoost only — lean, no GPU needed
pip install -e ".[tabicl]"        # TabICL foundation model (pulls torch)
pip install -e ".[uncertainty]"   # conformal prediction
pip install -e ".[all]"           # everything
```

**2. Decide the TabICL question now, not at runtime.**

TabICL downloads a pretrained checkpoint (a few hundred MB) on its first `fit`, and
wants a GPU above roughly 10,000 rows. If the target machine is air-gapped, is
CPU-only, or has no room for the checkpoint, install without the `tabicl` extra.
`tabular-tabicl` detects the missing package, reports it, and the orchestrator falls
back to CatBoost. That is a clean degradation — but it should be your decision, made
here, rather than a surprise mid-analysis.

**3. Verify the pack is intact.**

```bash
python scripts/validate_skills.py
```

Checks that every skill has valid frontmatter with `name` and `description`, that
folder names match declared names, that every `tabular-*` reference in the
orchestrator resolves, and that `defaults.yaml` parses. Run this after editing any
skill and before publishing the repo.

**4. Optionally, seed a project yourself.**

If you would rather hand an agent a project that already has the skills in place:

```bash
python scripts/install_skills.py add --all --target /path/to/project
python scripts/install_skills.py add --core --target /path/to/project   # lean 7-skill set
python scripts/install_skills.py list --target /path/to/project
python scripts/install_skills.py remove --all --target /path/to/project
```

Supports `--link` to symlink instead of copy (upstream edits propagate) and
`--dry-run` to preview. This script is **ergonomics, not architecture** — it does
nothing an agent could not do with `cp -r`. Nothing in the pack depends on it.

**5. Tune policy, not prose.**

`skills/tabular/defaults.yaml` holds every number the orchestrator relies on: dataset
size bands, CV folds, metric choices, the ensemble adoption threshold, time budgets.
Edit values there to change agent behavior across the whole pack. This is the
intended customization surface.

---

### For Agents / Skill Deployment

Hermes installs these skills on its own. No human is in this loop.

**Add skills to the current project.** They are plain directories of Markdown; a
copy is the whole operation.

```bash
mkdir -p .hermes/skills
cp -r /path/to/hermes-tabular/skills/* .hermes/skills/
```

Add a subset the same way — but always include `tabular`, which is the entry point,
and the skills it will delegate to:

```bash
cp -r /path/to/hermes-tabular/skills/{tabular,tabular-inspect,tabular-load,tabular-split,tabular-baseline,tabular-catboost,tabular-evaluate} .hermes/skills/
```

**Remove skills.**

```bash
rm -rf .hermes/skills/tabular-tabicl        # drop one delegate
rm -rf .hermes/skills/tabular*              # drop the whole pack
```

Note the glob: `tabular-*` matches the nine delegates but **not** the `tabular`
orchestrator itself, leaving a stranded entry point. Use `tabular*` to remove
everything.

**Deployment contract**

- Skills live at `.hermes/skills/<skill-name>/`, each containing a `SKILL.md`.
- Installation is **idempotent** — re-copying over an existing install is a safe
  update. Removal is a plain `rm -rf`, leaving no residue elsewhere in the project.
- Skills carry **no state and no imports**. Nothing registers, nothing initializes,
  no file outside `.hermes/skills/` is touched. Adding or removing one cannot break
  another.
- The only cross-skill coupling is that `tabular` delegates by name. If a delegate
  is absent, that path simply does not resolve; the other nine still work.
- `scripts/install_skills.py` is available if you prefer it — it is idempotent and
  warns about dangling delegations — but it is optional. Plain `cp`/`rm` is fewer
  moving parts and one less dependency.

**Then invoke exactly one skill: `tabular`.**

The other nine are delegation targets. Do not invoke them directly; the orchestrator
decides which are needed, in what order, and reads `defaults.yaml` for thresholds.

**Before modeling, read `references/pitfalls.md`.** Leakage, temporal and group
leakage, preprocessing fit before splitting, and test-set reuse all produce
excellent validation numbers and worthless models. The checklist at the bottom of
that file is the fastest way to avoid shipping a confidently wrong result.

**Runtime expectations**

- Missing optional dependency (`catboost`, `tabicl`, `mapie`) → report it and
  continue with what is installed. Never silently skip a model family.
- Skills are documentation, not a library. There is nothing to import; the reference
  implementations in each `SKILL.md` are meant to be adapted to the dataset at hand,
  not called as a fixed API.

---

## Usage

Invoke only the orchestrator:

> Use the `tabular` skill on `data/customers.csv`. Target is `churned`. I need
> calibrated probabilities and an honest estimate of out-of-sample AUC.

`tabular` inspects, plans, and delegates. Everything else is an implementation
detail it calls on your behalf.

---

## Design notes

- **The orchestrator holds policy, not code.** `skills/tabular/SKILL.md` contains
  almost no implementation detail. Every other skill is focused, testable, and
  replaceable.
- **Swapability.** If TabICL is superseded, only `tabular-tabicl` changes. The
  orchestration policy stays stable.
- **Tunable thresholds.** `defaults.yaml` holds the numbers so policy prose stays
  stable while values evolve.
- **Nothing to import.** The pack is Markdown. That is what makes it trivially
  installable and removable by an agent, and what keeps the human's job confined to
  the Python environment.
- **Future harnesses.** Hermes is the primary supported harness. If others are ever
  needed, add a thin `adapters/` layer rather than renaming the project.

---

## License

MIT. See [LICENSE](LICENSE).

## Attribution & Disclaimer

This project is an independent community package built for use with the [Hermes Agent](https://github.com/nousresearch/hermes-agent) framework. It is **not** an official repository maintained or endorsed by Nous Research.   

For the core agent framework, documentation, and official tools, visit [Nous Research](https://nousresearch.com).