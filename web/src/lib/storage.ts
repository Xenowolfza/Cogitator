import { FACTIONS } from "../data/factions";
import { uid } from "./id";
import type { AppStore, ArmyList } from "./types";

const KEY = "wh40k-strike-force-v1";

export function emptyList(partial?: Partial<ArmyList>): ArmyList {
  const faction = FACTIONS[0];
  return {
    id: uid(),
    name: "New Strike Force",
    factionId: faction.id,
    detachmentId: faction.detachments[0].id,
    pointsCap: 2000,
    selections: [],
    updatedAt: new Date().toISOString(),
    ...partial,
  };
}

export function defaultStore(): AppStore {
  const list = emptyList();
  return { lists: [list], activeId: list.id };
}

function isList(value: unknown): value is ArmyList {
  if (!value || typeof value !== "object") return false;
  const v = value as Record<string, unknown>;
  return (
    typeof v.id === "string" &&
    typeof v.name === "string" &&
    typeof v.factionId === "string" &&
    typeof v.detachmentId === "string" &&
    typeof v.pointsCap === "number" &&
    Array.isArray(v.selections)
  );
}

export function loadStore(): AppStore {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return defaultStore();
    const parsed = JSON.parse(raw) as AppStore;
    const lists = (parsed?.lists ?? []).filter(isList);
    if (!lists.length) return defaultStore();
    const active = lists.find((l) => l.id === parsed.activeId) ?? lists[0];
    return { lists, activeId: active.id };
  } catch {
    return defaultStore();
  }
}

export function saveStore(store: AppStore): void {
  localStorage.setItem(KEY, JSON.stringify(store));
}

export function exportList(list: ArmyList): string {
  return JSON.stringify(list, null, 2);
}

export function parseImportedList(text: string): ArmyList {
  const parsed: unknown = JSON.parse(text);
  if (!isList(parsed)) throw new Error("That file is not a Strike Force list.");
  return {
    ...parsed,
    id: uid(),
    updatedAt: new Date().toISOString(),
  };
}
