# Privacy Plan — Home Recordings

Last updated: 2026-11-01.

## What is recorded

Phone or Jetson camera video of the prep zone (cutting board and bowls) during cooking sessions, plus tap events and corrections.

## Rules

1. **Local only.** Recordings stay on the Mac or the Jetson, in `sessions/`. Nothing is uploaded.
2. **Never in git.** `.gitignore` excludes `sessions/`, `data/raw/`, `data/cache/`, `*.mp4`, and `*.avi`.
3. **No third-party training.** No recording goes to an external model or service. The Cosmos teacher model runs locally (Jetson or Mac). Any remote service needs a separate, consolidated approval first (account, data sent, retention, cost, fallback).
4. **Frame the prep zone only.** Point the camera at the board and bowls. Avoid faces, other rooms, and screens.
5. **Public materials use public data only.** README, demo, and CI use Breakfast Actions clips or synthetic data, never home recordings, unless the user approves a specific clip.

## Deletion

- One session: delete `sessions/<session_id>/`. Derived files (embeddings, AI labels) carry the session ID and are deleted with `kcp delete-session <session_id>` (built when personal sessions start, in Phase 3).
- Everything: delete `sessions/` and `data/cache/`.
