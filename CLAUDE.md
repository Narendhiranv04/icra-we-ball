# Claude entry point

Read [`CLAUDE_HANDOFF.md`](CLAUDE_HANDOFF.md) completely before modifying this
repository, **starting with its section 0**, which carries the current state and
supersedes older status claims elsewhere in the file. It contains the research
intent, architecture boundaries, experimental protocols, verified limitations,
remote-model setup, decisions that must not be silently reverted, and
prioritized next steps.

On a machine that has not run this repository before, read
[`MACHINE_HANDOFF.md`](MACHINE_HANDOFF.md) first: a clone is not sufficient to
run anything. `runs/`, the virtualenv, the PDDLStream checkout and the
semantic-model weights all live outside version control.

Before reporting or changing anything about the baselines, read
[`BASELINE_FIDELITY.md`](BASELINE_FIDELITY.md). It defines what may be claimed,
the decoding conditions, and the physical-execution controls that are part of
the reported result definition.

Also inspect the live worktree before acting:

```bash
git status --short --branch
git log -5 --oneline --decorate
```

Run the test suite as `env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
.venv/bin/python -m pytest ...`. With ROS sourced, its pytest plugin hijacks
collection and the suite exits 0 having tested nothing. Expect exactly seven
failures, all Kitchen or robot-profile; any other failure is real.

Do not reset, clean, discard, or overwrite the existing worktree.
