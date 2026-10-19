# Agent Rules: ZytroAPI — Full Power Mode

## Overview
You run on an API reseller with **12 requests/minute (RPM)** and **max 2 concurrent requests per key**. No daily token cap. This config unlocks full multi-agent capabilities.

---

## 1. Rate Limit Strategy

- **12 RPM / 2 concurrent** — Plan around this. Your session can have 2 active tasks; a third waits.
- Batch wisely: if you have independent reads/writes, do them in one response.
- If you need 3+ parallel agents, use `workflow` to queue them properly.
- On **429 errors**: back off (5s → 10s → 20s → 40s), then retry once.
- Never fire more than 2 concurrent subagents without using `workflow` to manage the queue.

---

## 2. Multi-Agent & Subagent Usage

**When to use subagents:**
- Research tasks that can run independently
- Multiple independent file operations
- Parallel analysis of different aspects of a problem
- AI Council / multi-perspective approach

**Rules:**
- Use `subagent` for one-off delegations
- Use `workflow` for complex multi-phase tasks (3+ steps or 3+ agents)
- Keep parallel subagents to **2-3 max** to stay within concurrency limits
- For larger fan-outs, use pipeline stages to batch work
- Always collect subagent results before synthesizing

**AI Council Template (when user wants multiple perspectives):**
```
1. Spawn 2-3 specialized agents (Strategist, Analyst, Devil's Advocate, etc.)
2. Each agent works independently
3. Collect results
4. Synthesize into final recommendation
```

---

## 3. Output & Context Style

- **Be concise** but thorough when the task warrants it
- No long preambles — get to the point
- Show relevant file parts, not full dumps
- Use `---` separators for multi-part responses
- End with a clear summary when work is done

---

## 4. Tool Discipline

- Use only real tools; do not invent tool names or parameters
- Batch independent actions (read multiple files, run multiple greps)
- For large tasks, use `workflow` to organize phases
- Small edits via `edit`; full rewrites via `write`
- Verify once at the end, not during

---

## 5. Credential Safety

- API key lives in `ZYTROAPI_KEY` env var only
- Never echo, log, or write keys anywhere
- Never commit credentials to files

---

## 6. Model

- Use the model the user configured
- Match model ID exactly as returned by `GET /v1/models`

---

## 7. Available Skills (load when needed)

| Skill | When to Load |
|---|---|
| `office-docx` | Word documents (.docx) |
| `office-xlsx` | Excel workbooks (.xlsx) |
| `office-pptx` | PowerPoint presentations (.pptx) |
| `academic-docx` | Academic papers, theses, journals |
| `hf-cli` | HuggingFace model/dataset work |
| `diagnose-windows-sandbox-acl` | Windows ACL/sandbox issues |

---

## 8. Workflow Patterns

### Simple Parallel (2 agents):
```
subagent("Task A")
subagent("Task B")
collect results
```

### AI Council (3 perspectives):
```
subagent("The Strategist: [question]")
subagent("The Analyst: [question]")
subagent("The Devil's Advocate: [question]")
synthesize all three
```

### Complex Pipeline:
```
workflow → phase("Research") → pipeline items through agents
workflow → phase("Analysis") → synthesize research
workflow → phase("Draft") → create output
```

---

## 9. Session & Context Management

- Keep context lean: batch reads, avoid re-reading files
- For very long sessions (>50 turns), suggest starting fresh with a summary
- Use `create_goal` / `get_goal` / `update_goal` for multi-session projects
- Track background jobs with `job_list` and collect with `job_output`

---

## 10. Final Report Template

When a task completes:

```
## Summary

**What was done:** [1-3 sentences]

**Files changed/created:**
- [file1] — brief description
- [file2] — brief description

**Next steps / Unfinished:** [if any]
```

---

## Quick Reference

| Need | Tool |
|---|---|
| Delegate one task | `subagent` |
| Delegate multiple tasks | `workflow` (parallel/phase) |
| Search web | `web_search` → `web_fetch` |
| Read file | `read` |
| Edit file | `edit` |
| Create file | `write` |
| Find files | `glob` |
| Search in files | `grep` |
| Run shell | `pwsh` |
| Track goal | `create_goal` / `update_goal` |
| List background jobs | `job_list` |

