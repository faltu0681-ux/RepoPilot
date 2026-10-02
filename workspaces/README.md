# Repository workspaces

Drop a repository here, one folder per repository, then start a run against it with
`local://<folder-name>`:

```
workspaces/
├── project-a/
├── project-b/
└── project-c/
```

RepoPilot never writes to these folders. Each run copies the one it was pointed at into its own
per-run checkout under `.repopilot-data/` and works on that copy, so your files are unchanged
afterwards no matter how far the agent goes.

This directory is configured by `REPOPILOT_REPOSITORIES_DIR` and is gitignored, because its
contents are your code rather than RepoPilot's. Set the variable to move it anywhere else.