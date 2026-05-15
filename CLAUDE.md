<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **rinkuzu-ai-api** (4306 symbols, 7551 relationships, 175 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/rinkuzu-ai-api/context` | Codebase overview, check index freshness |
| `gitnexus://repo/rinkuzu-ai-api/clusters` | All functional areas |
| `gitnexus://repo/rinkuzu-ai-api/processes` | All execution flows |
| `gitnexus://repo/rinkuzu-ai-api/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |
| Work in the Learning area (114 symbols) | `.claude/skills/generated/learning/SKILL.md` |
| Work in the Content_pipeline area (97 symbols) | `.claude/skills/generated/content-pipeline/SKILL.md` |
| Work in the Api area (59 symbols) | `.claude/skills/generated/api/SKILL.md` |
| Work in the Routers area (47 symbols) | `.claude/skills/generated/routers/SKILL.md` |
| Work in the Quiz area (46 symbols) | `.claude/skills/generated/quiz/SKILL.md` |
| Work in the Persistence area (33 symbols) | `.claude/skills/generated/persistence/SKILL.md` |
| Work in the Stages area (28 symbols) | `.claude/skills/generated/stages/SKILL.md` |
| Work in the Graph area (20 symbols) | `.claude/skills/generated/graph/SKILL.md` |
| Work in the Llm area (20 symbols) | `.claude/skills/generated/llm/SKILL.md` |
| Work in the Prompts area (16 symbols) | `.claude/skills/generated/prompts/SKILL.md` |
| Work in the Tests area (14 symbols) | `.claude/skills/generated/tests/SKILL.md` |
| Work in the Application area (12 symbols) | `.claude/skills/generated/application/SKILL.md` |
| Work in the Embed area (10 symbols) | `.claude/skills/generated/embed/SKILL.md` |
| Work in the Merge area (8 symbols) | `.claude/skills/generated/merge/SKILL.md` |
| Work in the Chunkers area (7 symbols) | `.claude/skills/generated/chunkers/SKILL.md` |
| Work in the Storage area (7 symbols) | `.claude/skills/generated/storage/SKILL.md` |
| Work in the Scripts area (5 symbols) | `.claude/skills/generated/scripts/SKILL.md` |
| Work in the Schemas area (4 symbols) | `.claude/skills/generated/schemas/SKILL.md` |

<!-- gitnexus:end -->
