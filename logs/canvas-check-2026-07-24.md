# Canvas check — 2026-07-24 (scheduled task report)

## Outcome: could not check Canvas or update Notion

Two blockers stopped the scheduled run:

1. **Claude in Chrome extension is not connected.** Canvas (kth.instructure.com/courses/8199) sits behind KTH SSO, so it can only be accessed through a signed-in browser session. The Chrome extension is the only tool in this session that can do that, and it reported "not connected" when the run started.
2. **Notion MCP is not authorized.** The Notion connector requires an OAuth flow that can only run in an interactive session. It came back as unauthorized, so no page could be created or updated.

## What I could see (public Kurswebb only)

The public course pages at kth.se/social/course/DM128X show:

- Kursöversikt: "Ingen aktivitet senaste månaden" (no activity in the last month).
- Nyhetsflöde: no public entries.
- Schema: "Inga händelser matchade sökningen" — no scheduled events for the current period.

None of this reflects Canvas announcements, assignments, or files, which are only visible after login. So this is not evidence that there are no new updates — just that the public surface is quiet.

## To make future scheduled runs work

- Install / sign in to the Claude in Chrome extension: https://chromewebstore.google.com/detail/fcoeoabgfenejglbffodgkkbkcdhcgfn — then keep Chrome open and signed into KTH so the scheduled run can reach Canvas.
- Authorize the Notion connector via claude.ai → connector settings (or `/mcp` in an interactive session).

Once both are connected, the same scheduled task should be able to log into Canvas, scan Announcements / Modules / Assignments / Files for anything newer than the last run, and write a summary to the relevant Notion page.

## Links

- Canvas course (login required): https://kth.instructure.com/courses/8199
- Public Kurswebb: https://www.kth.se/social/course/DM128X/
- Kurs-PM: https://www.kth.se/kurs-pm/DM128X/
