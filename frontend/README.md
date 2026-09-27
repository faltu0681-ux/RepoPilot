# RepoPilot frontend

The terminal UI. Start the backend on port 8000 first, then:

```bash
npm install
npm run dev
```

Open <http://localhost:5173>. The dev server proxies `/api` and `/ws` to the backend, so the
browser only ever talks to one origin.

```bash
npm run lint      # oxlint
npm run build     # tsc -b && vite build
npm run preview   # serve the production build
```

| File | Role |
| --- | --- |
| `src/types.ts` | Wire types, mirroring `backend/repopilot/domain/models.py` and `api/ws.py` |
| `src/api.ts` | REST client for creating, reading, and cancelling runs |
| `src/useRunStream.ts` | WebSocket subscription: cursor, resync, and reconnect handling |
| `src/App.tsx` | The form and the live terminal |

Set `REPOPILOT_BACKEND` to point the proxy at a backend on a different address or port.
