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

1. Pick a faction (Imperium, Chaos, or Xenos) and a detachment. Built-in detachments are short playstyle notes using public 11th-edition names. Official rule text is not included. Use **Your detachment notes** to type your own summary, enhancements, or attach a faction-pack PDF that stays in this browser.
2. Add units from the catalogue (or **Custom unit** for anything missing). Chapter armies also include shared Adeptus Astartes units.
3. Set model counts, wargear, and character enhancements. **Edit stats** on any list entry to override M/T/SV, weapons, or abilities for that copy.
4. Open **Datasheets** to extract a card for every unit. Use **Card colour** to switch parchment, blood, waaagh, ultramarine, gold, night, necrodermis, or ivory.
5. **Print pack** (browser print → Save as PDF) or **Export JSON** to move a list between browsers.

Lists are stored in this browser (`localStorage`).

## Build

```bash
npm run build
```
