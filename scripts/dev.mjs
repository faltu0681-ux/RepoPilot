/**
 * Run the API and the web app together, in one terminal.
 *
 * A dependency-free stand-in for a task runner, and deliberately so. The only thing worth
 * automating here is "start these two processes and shut both down together", which is a few
 * dozen lines. Pulling in Turborepo (or `concurrently`) to do that would add a toolchain whose
 * main job would be shelling out to `uvicorn` and `vite`, and whose dependency graph would be
 * empty anyway: `apps/api` and `apps/web` share no code.
 *
 * The API side runs as a plain Python command rather than as an npm workspace. Its dependencies
 * are pip's business, resolved into the virtualenv, and wrapping them in package.json metadata
 * would only create a second place to keep them in sync.
 *
 *     npm run dev
 */

import { spawn } from 'node:child_process'
import { existsSync } from 'node:fs'

/** Windows needs the shell to resolve `npm`, and a Unix npm to resolve `python`. */
const IS_WINDOWS = process.platform === 'win32'

/**
 * The Python interpreter to use: the project's virtualenv when there is one.
 *
 * Anything else is left as a bare name so the developer's own environment decides. Hardcoding a
 * path would break on macOS and Linux, where the layout is `bin/` rather than `Scripts/`, and it
 * would ignore a deliberately activated environment on a machine without `.venv` at all.
 */
function pythonExecutable() {
  if (process.env.REPOPILOT_PYTHON) return process.env.REPOPILOT_PYTHON
  const inVenv = IS_WINDOWS ? '.venv/Scripts/python.exe' : '.venv/bin/python'
  if (existsSync(inVenv)) return inVenv
  return IS_WINDOWS ? 'python' : 'python3'
}

const processes = [
  {
    name: 'api',
    command: pythonExecutable(),
    args: ['-m', 'repopilot.cli', 'serve'],
    shell: false,
  },
  {
    name: 'web',
    command: 'npm',
    args: ['run', 'dev', '--workspace', '@repopilot/web'],
    shell: IS_WINDOWS,
  },
]

let shuttingDown = false

function shutdown(code) {
  if (shuttingDown) return
  shuttingDown = true
  for (const child of children) killTree(child)
  process.exitCode = code
}

/**
 * On Windows a child process is killed with TerminateProcess, which does not reach the reloader
 * or the Vite child it spawned. Without taskkill on the tree, closing this terminal leaves orphan
 * servers holding port 8000 and 5173, which is the sort of thing that makes the next run fail for
 * no visible reason.
 */
function killTree(child) {
  if (child.exitCode !== null) return
  if (!IS_WINDOWS) {
    child.kill()
    return
  }
  spawn('taskkill', ['/pid', String(child.pid), '/T', '/F'], { stdio: 'ignore' })
}

const children = processes.map(({ name, command, args, shell }) => {
  const child = spawn(command, args, {
    shell,
    stdio: ['ignore', 'inherit', 'inherit'],
    env: process.env,
  })

  child.on('error', (error) => {
    console.error(`[dev] ${name} failed to start: ${error.message}`)
    shutdown(1)
  })

  child.on('exit', (code, signal) => {
    if (shuttingDown) return
    // One side dying takes the other down: a dev session with a dead API and a live UI that only
    // shows a socket error is harder to diagnose than one that stopped.
    console.error(`[dev] ${name} exited (code ${code ?? 'null'}, signal ${signal ?? 'none'})`)
    shutdown(code ?? 1)
  })

  return child
})

for (const signal of ['SIGINT', 'SIGTERM']) {
  process.on(signal, () => shutdown(0))
}

console.log('[dev] api -> http://127.0.0.1:8000')
console.log('[dev] web -> http://localhost:5173')