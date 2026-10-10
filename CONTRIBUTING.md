# Contributing — C3 Explainable RAG Knowledge Assistant & Singlish Retrieval Engine

This Python/FastAPI backend is Component 3 (C3). The team's Next.js frontend is a separate shared repository, and frontend changes go there through a reviewed Pull Request.

## Branches

Format: `type/c3-short-feature-name` (lowercase, hyphens, short).

| Type | Use for |
|---|---|
| `feat` | New capability |
| `fix` | Bug fix |
| `refactor` | Restructure without behaviour change |
| `test` | Tests only |
| `research` | Experiments, evaluation, ablations |
| `docs` | Documentation only |
| `chore` | Tooling, dependencies, CI |

Backend examples: `feat/c3-fastapi-service`, `feat/c3-retrieval-api`, `feat/c3-creator-channel-scope`, `feat/c3-qdrant-integration`, `feat/c3-sinhala-rag`, `feat/c3-thread-aware-rag`, `feat/c3-claim-verification`, `research/c3-retrieval-ablation`, `research/c3-human-benchmark`, `fix/c3-singlish-normalization`.

Frontend repository prefixes by component: `c1` Market Research, `c2` Video Prediction, `c3` Knowledge Assistant, `c4` Analytics. For example: `feat/c3-knowledge-assistant-ui`, `feat/c3-video-selector`, `feat/c3-retrieval-api-integration`, `feat/c3-evidence-display`.

Rules:
- Create a branch only when its work starts. Keep it short-lived.
- `main` and `development` are protected: no direct pushes, no force-pushes, changes via Pull Request only.
- **Feature branches target `development`** (integration branch). `development` is merged into `main` through a reviewed release PR, and only when it is stable and CI is green.
- Branch from the latest `development`.
- Never rewrite shared history (`push --force`, rebasing pushed branches) and never delete or rename other people's branches.

## Commits

[Conventional Commits](https://www.conventionalcommits.org/) style, with the type matching the branch type:

```
feat(api): add POST /v1/retrieve
fix(normalization): keep unresolved tokens in Latin script
research(eval): add cross-script ablation results
```

- Imperative mood, subject line of 72 characters or less, explain *why* in the body when it is not obvious.
- One logical change per commit, and keep the tests passing at every commit.
- Never commit secrets (`.env`), raw data containing personal identifiers, model weights or large derived indexes. Check `git status` / `git diff --staged` before committing.

## Pull Requests

- Small and focused, and link the related issue or milestone.
- The description covers what changed, why, how it was tested, and any API contract or data changes.
- CI must pass: `python -m unittest discover -s tests -t .` and `python scripts/export_openapi.py --check`.
- **API contract changes** (`docs/openapi.json`): adding optional fields is fine. Renaming or removing fields, or changing an enum, needs a new version (`/v2`) and a heads-up to the frontend team.
- **Research results** (`results/`): never overwrite reported numbers silently. Explain regenerated results in the PR.
- Squash-merge or merge after review, then delete the merged feature branch.

## Code review

- At least one reviewer approval before merging.
- Reviewers check correctness, tests, privacy and data handling, contract stability, and honest claims. Results must not be presented as validated when they are not.
- Authors respond to every comment. Resolve threads only when the comment is addressed or agreed.

## Data privacy

The raw YouTube export (`data/youtube_comments_multiple.csv`) contains public author handles. It is git-ignored and must stay local and private. Only the author-stripped `data/comments_clean.csv` may be considered for version control, subject to ethics approval.

**Known history issue:** an earlier backend repository's history contained the raw CSV with author handles. That history must not be pushed, mirrored or transferred into this or any other public repository. Do not try to "fix" it by rewriting shared history without team agreement. Any new repository starts with clean history.
