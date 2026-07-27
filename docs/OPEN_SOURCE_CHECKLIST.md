# Open Source Release Checklist

## Required Before Public Release

- [x] Remove private notes and internal agent logs from the publishable tree.
- [x] Add root README, license, notice, security, and contribution docs.
- [x] Align example environment and Docker defaults with the Ollama-only runtime.
- [ ] Rotate any credential that previously appeared in Git history.
- [ ] Rewrite or recreate public history so deleted private files are not recoverable from commits.
- [ ] Re-run backend tests from a clean checkout.
- [ ] Confirm repository visibility, topics, and release notes before publishing.

## Ongoing Hygiene

- Keep `.env`, datasets, generated reports, model artifacts, and local logs ignored.
- Prefer small fixtures over checked-in real data.
- Document any required paid service before introducing it; this project defaults to a zero-budget deployment policy.
