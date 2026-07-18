# AGENTS.md

## General commands

- No setup needed. Do not run "cd", assume you are in the correct path by default.
- Ensure `GI_TYPELIB_PATH` is set before running the app or tests — the `rayforge` and `test` pixi tasks handle this automatically via `pkgconf`.
- Translations must be compiled before running tests or the app:
  ```
  pixi run compile-translations
  ```
- Key pixi commands:
  - `pixi run lint`: Runs flake8 + pyflakes + pyright (sequentially, via `depends-on`)
  - `pixi run format`: Auto-format with `ruff format rayforge tests scripts`
  - `pixi run test`: Run backend tests (skips `ui` and `stress` markers)
  - `pixi run uitest`: Run only UI tests (`-m ui`)
  - `pixi run rayforge`: Launch the app (wraps with `with_gdk.sh`)
  - `pixi run testapp`: Launch the app with extended GI_TYPELIB_PATH for debugging
  - `pixi run wheel`: Build a PyPI wheel (depends-on `compile-translations`)
  - `pixi run clean`: Clean build artifacts
  - `pixi run pre-commit-install`: Install the pre-commit hook (run once per clone)
  - `pixi run print-untranslated list`: List languages with missing translations
  - `pixi run print-untranslated <lang>`: Show missing translation keys
- Use `pixi run <task>` — the project uses `pixi` not `pip` or `poetry`.
- There are four pixi environments: `default` (dev), `website`, `build`, `video`.
  Switch with `pixi run -e <env> <task>`.

## Linting & code style

- Lint first, test later. `pixi run lint` bundles three checks that run **sequentially** (via `depends-on`). If flake fails, pyflakes and pyright are skipped until flake passes:
  1. `flake8` — ignores E127,E128,E121,E123,E126,E203,E226,E24,E704,W503,W504; builtins=`_`
  2. `pyflakes` — env `PYFLAKES_BUILTINS=_`
  3. `pyright` — stub path is `typings`
- Pre-commit hook runs `pixi run lint` only. Install it with `pixi run pre-commit-install`.
- Python max line length: **79 characters** (enforced by ruff format + flake8).
- Ruff format uses **double quotes** (set in pyproject.toml).
- PEP8 compliance expected. Keep functions small and testable.
- Retain existing formatting, docstrings, and comments.
- Never mark changes with inline comments; code is for clean final implementation only.
- The `_` function from gettext is a builtin (configured in both flake8 and pyflakes).

## Testing

### Markers
Three test markers are defined in `pyproject.toml`:
- `ui`: Requires a graphical environment / GTK event loop (runs via `xvfb-run` in CI)
- `stress`: Long-running stress tests
- (default): Everything else — backend logic tests

Default `pytest` config excludes `ui` and `stress`: `addopts = "-m 'not ui and not stress'"`.

### Run a single test
```
pixi run test tests/core/test_something.py -k "test_name"
```

### Run a single test package
```
pixi run test tests/image/svg/
```

### Run UI tests
```
pixi run uitest          # all UI tests
pixi run uitest -k test_name
```
UI tests need a display (use `xvfb-run pixi run uitest` on a headless system).

### Test structure
Tests live under `tests/` mirroring the `rayforge/` package layout:
- `tests/core/`, `tests/image/`, `tests/machine/`, `tests/pipeline/`, etc.
- `tests/ui_gtk/` exists but is excluded from `pixi run test` via `--ignore=tests/ui_gtk`.
- Addon tests live under `rayforge/builtin_addons/*/tests/` and `rayforge/private_addons/*/tests/`.

### Conftest guarantees (you can rely on these)
- `RAYFORGE_CONFIG_DIR` is set to an isolated temp dir before any rayforge imports.
- `GLib.idle_add` is patched in backend (non-UI) tests — calling it from a test will `pytest.fail()`.
- The `RayforgeContext` singleton is fully shut down and reset after every test (`clean_context_singleton` autouse fixture).
- `multiprocessing` uses `spawn` start method on Linux (set in pytest_configure).
- pyvips debug logging is suppressed to WARNING+ level by default.

### Fixture hierarchy (choose the lightest one that works)
| Fixture | Scope | When to use |
|---|---|---|
| `isolated_context` / `isolated_machine` | function | Pure math tests, zero I/O, pure mocks. Fastest. |
| `lite_context` + `machine` / `sync_machine` | function | Machine tests without addons/cameras/materials. |
| `context_initializer` + `doc_editor` | function async | Full integration tests with addons loaded. |
| `ui_context_initializer` / `ui_task_mgr` | function sync | UI tests with real GLib event loop. |
| `mock_task_mgr` | function | Pipeline tests — synchronous mock, no async overhead. |

### Async test fixtures
- `task_mgr` (async fixture): Isolated TaskManager for async tests.
- `context_initializer` (async fixture): Full context with config isolation.
- Use `pytest.mark.asyncio` (strict mode: `asyncio_mode = "strict"` means tests must use `pytest.mark.asyncio`).
- **`asyncio_default_fixture_loop_scope = "function"`** is set in pyproject.toml. Async fixtures default to function-scoped event loops. Use `loop_scope="session"` explicitly for session-scoped async fixtures.

## Architecture

### Entry point
- **GUI entry**: `rayforge.app:main` — registered as `[project.gui-scripts]` in pyproject.toml.
- **CLI/headless**: `rayforge/app.py --exit --loglevel DEBUG` (imports files and exits).
- **CLI flags**: `--version`, `--vector`, `--trace`, `--exit`, `--uiscript <path>`, `--config <dir>`, `--loglevel`.

### Key directories
| Path | Purpose |
|---|---|
| `rayforge/app.py` | Application bootstrap, platform init, shutdown orchestration |
| `rayforge/core/` | Domain models (geometry, layers, materials, recipes, steps) |
| `rayforge/ui_gtk/` | GTK4/Libadwaita UI code |
| `rayforge/machine/` | Machine models, controllers, G-code dialects, serial/network I/O |
| `rayforge/pipeline/` | Pipeline engine: producers, transformers, artifact management |
| `rayforge/image/` | Image/SVG/PDF/DXF import/export, tracing |
| `rayforge/shared/` | Shared utilities: tasker (process pool), GLib helpers |
| `rayforge/doceditor/` | Document editor (undo, file operations) |
| `rayforge/addon_mgr/` | Addon loading and management |
| `rayforge/context.py` | Global application context singleton |
| `rayforge/config.py` | Configuration paths and settings |
| `rayforge/camera/` | USB camera integration and calibration |
| `rayforge/simulator/` | 3D toolpath simulation |
| `rayforge/workbench/` | Workbench UI panels |
| `rayforge/undo/` | Undo/redo system |
| `rayforge/builtin_addons/` | Bundled addons (each has own `tests/` dir) |
| `rayforge/private_addons/` | Optional private addons |

### Versioning
- Version is auto-detected (in order): `version.txt` file → `git describe` → `importlib.metadata`.
- `setuptools-git-versioning` (a setuptools plugin) provides Git-based versioning.

## Translations (gettext / .po / .mo)
- Locale files: `rayforge/locale/*/LC_MESSAGES/rayforge.po`
- Update: `pixi run update-translations` (requires gettext >= 0.25)
- Compile only: `pixi run compile-translations` (needed before testing/running)
- Check missing: `pixi run print-untranslated <language_code>`
- The `_()` function is a builtin everywhere (see flake8/pyflakes config).

## Build & release

| Command | Produces |
|---|---|
| `pixi run wheel` | PyPI wheel (depends-on `compile-translations`) |
| `pixi run build-deb` | Binary .deb for local testing |
| `pixi run build-deb-source` | Source .deb for PPA upload |
| `pixi run compile-translations` | Prerequisite for all builds |

CI runs for pushes to `main` + tags, and PRs to any branch. Release tags trigger PyPI and Snap publishing.

## Raygeo (Rust/PyO3 geometry library)

Even though Raygeo is installed as a regular pip dependency, we own it. If the root
cause of an issue is in Raygeo, you should fix it there instead of building a
workaround.
Source repository: https://github.com/barebaric/raygeo

### Testing with a local Raygeo checkout

`scripts/pixi-raygeo.sh` wraps any pixi command with a
`dependency-override` that uses a local raygeo checkout. The project's
real `pixi.toml`/`pixi.lock` are never permanently modified.

```bash
ln -s /path/to/raygeo external/raygeo    # one-time symlink (external/ is gitignored)
scripts/pixi-raygeo.sh run rayforge      # run against local raygeo
scripts/pixi-raygeo.sh run test          # test against local raygeo
scripts/pixi-raygeo.sh shell             # activate a shell with local raygeo
```

After editing raygeo Rust or Python source, rebuild it with:

```bash
scripts/rebuild-raygeo.sh                # clear uv cache + rebuild raygeo
```

To go back to the PyPI raygeo, just use `pixi run rayforge` without the
wrapper (or any other pixi command).

## Other rules

- Never use "head" to filter CLI commands — this hides useful error messages.
- Fix all linter errors before running tests. Do not run the full test suite prematurely.
- Use proper markdown: each file in a separate code block; file start markers go OUTSIDE code blocks.
- Do not make changes unrelated to the current task.
- Never remove logging or debugging unless explicitly asked by the user.
- Do not repeat files in responses unless they have changes.

## Ruida Driver (ruidarpa) Development
- The Ruida driver (ruidarpa) in development is in rayforge/machine/driver/ruidarpa
- The Ruida driver in rayforge/machine/driver/ruida is a prototype and is for reference only. Never modify files in the rayforge/machine/driver/ruida directory tree.
- The ruidarpa driver depends upon external/ruida-protocol-analyzer which is maintained elsewhere. Do not modify files in the external/ruida-protocol-analyzer directory tree.
