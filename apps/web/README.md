# RepoPilot web

The terminal UI. The API runs on port 8000; the dev server proxies `/api` and `/ws` to it, so the
browser only ever talks to one origin.

This app is an npm workspace, so `npm install` runs from the repository root rather than here.

```bash
npm run dev          # from the repo root, or: npm run dev --workspace @repopilot/web
```

Open <http://localhost:5173>.

```bash
npm run dev        # vite with the API proxy
npm run lint       # oxlint
npm run typecheck  # tsc -b
npm run build      # tsc -b && vite build
npm run preview    # serve the production build
```

| File | Role |
| --- | --- |
| `src/api.ts` | REST client for creating, reading, and cancelling runs, and for listing `local://` repositories |
| `src/useRunStream.ts` | WebSocket subscription: cursor, resync, and reconnect handling |
| `src/App.tsx` | The form, the live terminal, and the run panels |
| `vite.config.ts` | Dev server and the `/api` + `/ws` proxy to the API |

The wire types are not here: they live in `@repopilot/contracts`, which mirrors
`apps/api/repopilot/domain/models.py` and `apps/api/repopilot/api/ws.py`.

If `workspaces/` holds folders, they are read from `GET /api/runs/repositories` once on load and
offered as buttons that fill the repository field with `local://<name>`. That call is best effort
by design, so the form still accepts anything you type.

Set `REPOPILOT_BACKEND` to point the proxy at an API on a different address or port.