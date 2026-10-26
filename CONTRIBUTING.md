# Contributing

How work gets from an issue to `main`. If something here is wrong or missing, open a Docs issue.

**Setup:** install [uv](https://docs.astral.sh/uv/), then `make setup`. Run `make check` before every push; `make format` fixes most lint failures.

## 1. Pick up an issue

- Start only issues that are **ready**: an owner, a written definition of done (DoD), and no open blockers. Spikes and experiments have a timebox and a stop rule instead of a DoD, because we don't know the answer yet.
- **WIP limit: 5 issues in progress across the team** (four pairs plus one buffer). If we're at 5, help finish something instead of starting something new.
- Move the issue to *In Progress* on the [board](https://github.com/orgs/Sandhi-Probe/projects/1) when you start.

## 2. Branch

Name it `<issue#>-<slug>` and branch from an up-to-date `main`, e.g. `22-pair-schema`. One issue per branch.

## 3. Open a PR

- Start the description with `Closes #N` so the issue closes on merge.
- Say how each DoD item is met. The PR template asks for this.
- Small and early beats big and late. Open a draft PR if you want feedback before it's finished.

## 4. Review

- One approval to merge, from a code owner (`.github/CODEOWNERS`) once that file exists.
- Reviewers respond within **48 hours**. If you can't, say so and suggest someone else.
- Review for three things: does it meet the DoD, is it tested, would a new member understand it.
- `make check` must pass in CI.

## 5. Merge

Once it's approved and green, the author squash-merges and deletes the branch.

## The DoD is the merge bar

An issue is done when every DoD box is checked **and** the work is on `main`. If a box can't be met, agree in the issue to drop or move it. Don't merge around it.

## Time budgets, not estimates

An issue's due date is a budget, not a prediction. If work is going to run past it, say so in the issue early, then cut scope or drop it rather than quietly extending. (Borrowed from [Shape Up](https://basecamp.com/shapeup/1.2-chapter-03). It's one company's practice, not a proven method, but it suits a volunteer team where velocity can't be measured.)

## Spikes and experiments

Open these with the Spike or Experiment template: question, timebox, stop rule. When the timebox runs out, post what you found, even if it's "inconclusive", and close the issue. A negative result is still a result.

## Blockers go in the issue, not DMs

If you're stuck or waiting on someone, comment on the issue and @mention them. The blocker stays visible on the board, and the next person with the same question can find the answer.

## Notebooks

Notebooks are for exploring. Before a result is cited (at a sync, in an ADR, in the paper), move the code behind it into `src/sandhi_probe/` with a test. Name notebooks `<issue#>-<initials>-<topic>.ipynb` and clear outputs before committing.

## Decisions

Anything we don't want to re-argue gets an ADR in [`docs/adr/`](docs/adr/).
