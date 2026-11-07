import { useEffect, useMemo, useRef, useState } from "react";
import { CatalogPanel } from "./components/CatalogPanel";
import { DatasheetsView } from "./components/DatasheetsView";
import { ListPanel } from "./components/ListPanel";
import { UnitEditor } from "./components/UnitEditor";
import { FACTIONS, detachmentById, factionById } from "./data/factions";
import { CARD_THEMES } from "./data/themes";
import { uid } from "./lib/id";
import { listPoints, listValidity, profileFor } from "./lib/points";
import { resolveList } from "./lib/resolve";
import { emptyList, exportList, loadStore, parseImportedList, saveStore } from "./lib/storage";
import type { Allegiance, AppStore, ArmyList, CardThemeId, UnitProfile, UnitSelection } from "./lib/types";

const ALLEGIANCE_LABEL: Record<Allegiance, string> = {
  imperium: "Imperium",
  chaos: "Chaos",
  xenos: "Xenos",
};

type Tab = "builder" | "datasheets";

function patchActive(store: AppStore, patch: Partial<ArmyList>): AppStore {
  return {
    ...store,
    lists: store.lists.map((l) =>
      l.id === store.activeId ? { ...l, ...patch, updatedAt: new Date().toISOString() } : l,
    ),
  };
}

export default function App() {
  const [store, setStore] = useState<AppStore>(() => loadStore());
  const [tab, setTab] = useState<Tab>("builder");
  const [editor, setEditor] = useState<"create" | UnitSelection | null>(null);
  const [importError, setImportError] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);

  const list = store.lists.find((l) => l.id === store.activeId) ?? store.lists[0];
  const faction = factionById(list.factionId) ?? FACTIONS[0];
  const detachment = detachmentById(list.factionId, list.detachmentId);
  const pts = listPoints(list);
  const validity = listValidity(list);
  const resolved = useMemo(() => resolveList(list), [list]);
  const fill = Math.min(100, (pts / Math.max(list.pointsCap, 1)) * 100);

  useEffect(() => {
    saveStore(store);
  }, [store]);

  function setList(patch: Partial<ArmyList>) {
    setStore((s) => patchActive(s, patch));
  }

  function changeFaction(factionId: string) {
    const next = factionById(factionId) ?? FACTIONS[0];
    if (next.id === list.factionId) return;

    const custom = list.selections.filter((sel) => Boolean(sel.customProfile));
    const currentFactionUnits = list.selections.filter((sel) => !sel.customProfile);
    const stash = {
      ...(list.factionStash ?? {}),
      [list.factionId]: { detachmentId: list.detachmentId, selections: currentFactionUnits },
    };
    const restored = stash[next.id];
    const det =
      detachmentById(next.id, restored?.detachmentId ?? next.detachments[0].id) ?? next.detachments[0];
    const validEnh = new Set(det.enhancements.map((e) => e.id));
    const selections = [...(restored?.selections ?? []), ...custom].map((sel) =>
      sel.enhancementId && !validEnh.has(sel.enhancementId)
        ? { ...sel, enhancementId: undefined }
        : sel,
    );

    setList({
      factionId: next.id,
      detachmentId: det.id,
      selections,
      factionStash: stash,
    });
  }

  function addUnit(unit: UnitProfile) {
    setList({
      selections: [
        ...list.selections,
        {
          uid: uid(),
          unitId: unit.id,
          models: unit.minModels,
          optionIds: [],
          customProfile: unit.id.startsWith("custom-") ? unit : undefined,
        },
      ],
    });
  }

  function newList() {
    const created = emptyList({ name: "New Strike Force" });
    setStore((s) => ({ lists: [...s.lists, created], activeId: created.id }));
    setTab("builder");
  }

  function deleteList() {
    setStore((s) => {
      const remaining = s.lists.filter((l) => l.id !== s.activeId);
      const lists = remaining.length ? remaining : [emptyList()];
      return { lists, activeId: lists[0].id };
    });
  }

  function download() {
    const blob = new Blob([exportList(list)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${list.name.replace(/\s+/g, "-").toLowerCase()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function onImportFile(file: File) {
    setImportError("");
    file
      .text()
      .then((text) => {
        const imported = parseImportedList(text);
        setStore((s) => ({ lists: [...s.lists, imported], activeId: imported.id }));
        setTab("builder");
      })
      .catch(() => setImportError("Could not import that file."));
  }

  return (
    <div className="app">
      <header className="topbar no-print">
        <div className="brand">
          <p className="kicker">Unofficial fan tool</p>
          <h1>Strike Force</h1>
        </div>

        <div className="top-fields">
          <label>
            List
            <select
              value={list.id}
              onChange={(e) => setStore((s) => ({ ...s, activeId: e.target.value }))}
            >
              {store.lists.map((l) => (
                <option key={l.id} value={l.id}>
                  {l.name}
                </option>
              ))}
            </select>
          </label>
          <label className="grow">
            Name
            <input value={list.name} onChange={(e) => setList({ name: e.target.value })} />
          </label>
          <label>
            Faction
            <select value={list.factionId} onChange={(e) => changeFaction(e.target.value)}>
              {(["imperium", "chaos", "xenos"] as const).map((group) => (
                <optgroup key={group} label={ALLEGIANCE_LABEL[group]}>
                  {FACTIONS.filter((f) => f.allegiance === group).map((f) => (
                    <option key={f.id} value={f.id}>
                      {f.name}
                    </option>
                  ))}
                </optgroup>
              ))}
            </select>
          </label>
          <label>
            Detachment
            <select
              value={list.detachmentId}
              onChange={(e) => {
                const detachmentId = e.target.value;
                const next = detachmentById(list.factionId, detachmentId);
                const valid = new Set(next?.enhancements.map((enh) => enh.id) ?? []);
                setList({
                  detachmentId,
                  selections: list.selections.map((sel) =>
                    sel.enhancementId && !valid.has(sel.enhancementId)
                      ? { ...sel, enhancementId: undefined }
                      : sel,
                  ),
                });
              }}
            >
              {faction.detachments.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.name}
                </option>
              ))}
            </select>
          </label>
          <label>
            Points cap
            <select
              value={String(list.pointsCap)}
              onChange={(e) => {
                const v = e.target.value;
                if (v === "custom") {
                  const raw = window.prompt("Custom points cap", String(list.pointsCap));
                  const n = Number(raw);
                  if (Number.isFinite(n) && n > 0) setList({ pointsCap: Math.round(n) });
                  return;
                }
                setList({ pointsCap: Number(v) });
              }}
            >
              <option value="1000">1000</option>
              <option value="2000">2000</option>
              <option value="3000">3000</option>
              {![1000, 2000, 3000].includes(list.pointsCap) && (
                <option value={String(list.pointsCap)}>{list.pointsCap}</option>
              )}
              <option value="custom">Custom…</option>
            </select>
          </label>
        </div>

        <div className="pts-meter">
          <div className="pts-row">
            <strong>
              {pts} / {list.pointsCap}
            </strong>
            <span>pts</span>
          </div>
          <div className="bar">
            <i style={{ width: `${fill}%` }} className={validity.overCap ? "over" : undefined} />
          </div>
          {validity.overCap && <p className="warn">Over points cap</p>}
          {validity.empty && <p className="muted">Empty list</p>}
        </div>
      </header>

      <nav className="tabs no-print">
        <button type="button" className={tab === "builder" ? "tab on" : "tab"} onClick={() => setTab("builder")}>
          Builder
        </button>
        <button
          type="button"
          className={tab === "datasheets" ? "tab on" : "tab"}
          onClick={() => setTab("datasheets")}
        >
          Datasheets
        </button>
        <div className="tab-actions">
          <button type="button" className="btn ghost" onClick={newList}>
            New list
          </button>
          <button type="button" className="btn ghost" onClick={deleteList}>
            Delete list
          </button>
          <button type="button" className="btn ghost" onClick={download}>
            Export JSON
          </button>
          <button type="button" className="btn ghost" onClick={() => fileRef.current?.click()}>
            Import JSON
          </button>
          {tab === "datasheets" && (
            <>
              <label className="theme-picker">
                Card colour
                <span className="theme-swatches">
                  {CARD_THEMES.map((theme) => (
                    <button
                      key={theme.id}
                      type="button"
                      title={theme.label}
                      data-theme={theme.id}
                      className={(list.cardTheme ?? "parchment") === theme.id ? "theme-swatch on" : "theme-swatch"}
                      onClick={() => setList({ cardTheme: theme.id as CardThemeId })}
                    />
                  ))}
                </span>
              </label>
              <button type="button" className="btn gold" onClick={() => window.print()}>
                Print pack
              </button>
            </>
          )}
          <input
            ref={fileRef}
            type="file"
            accept="application/json"
            hidden
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) onImportFile(file);
              e.target.value = "";
            }}
          />
        </div>
      </nav>
      {importError && <p className="warn no-print">{importError}</p>}

      {detachment && (
        <p className="det-line no-print">
          <strong>{detachment.name}:</strong> {detachment.summary}
        </p>
      )}

      {tab === "builder" && (
        <main className="builder">
          {editor ? (
            <UnitEditor
              factionId={list.factionId}
              factionName={faction.name}
              existing={editor === "create" ? undefined : profileFor(editor)}
              onCancel={() => setEditor(null)}
              onSave={(profile) => {
                if (editor === "create") {
                  addUnit(profile);
                } else {
                  const models = Math.min(Math.max(editor.models, profile.minModels), profile.maxModels);
                  setList({
                    selections: list.selections.map((sel) =>
                      sel.uid === editor.uid ? { ...sel, customProfile: profile, models } : sel,
                    ),
                  });
                }
                setEditor(null);
              }}
            />
          ) : (
            <CatalogPanel factionId={list.factionId} onAdd={addUnit} onCustom={() => setEditor("create")} />
          )}
          <ListPanel
            list={list}
            profileFor={profileFor}
            onChange={(selections) => setList({ selections })}
            onEdit={(sel) => setEditor(sel)}
          />
        </main>
      )}

      {tab === "datasheets" && <DatasheetsView list={list} units={resolved} />}

      <footer className="disclaimer">
        Unofficial fan-made reference for personal use. Not affiliated with or endorsed by Games
        Workshop. Profiles are compact mechanical summaries you can override from your own books.
        Always check current official rules and points.
      </footer>
    </div>
  );
}
