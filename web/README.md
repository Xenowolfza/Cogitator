# Strike Force — 40k Army Builder

Unofficial fan-made Warhammer 40,000 army list builder with printable datasheet cards.

This is **not affiliated with Games Workshop**. Profiles are compact mechanical summaries for personal use. Check current official rules and points before play.

## Run

```bash
npm install
npm run dev
```

Open http://127.0.0.1:8501 (this repo’s forwarded app port). http://127.0.0.1:5173 also works when the proxy is running.

## Use

1. Pick a faction and detachment, set a points cap.
2. Add units from the catalogue (or **Custom unit** for anything missing).
3. Set model counts, wargear, and character enhancements.
4. Open **Datasheets** to extract a card for every unit in the list.
5. **Print pack** (browser print → Save as PDF) or **Export JSON** to move a list between browsers.

Lists are stored in this browser (`localStorage`).

## Build

```bash
npm run build
```
