import { detachmentById, extrasFor } from "../data/factions";
import { unitById } from "../data/catalog";
import type { ArmyList, UnitProfile, UnitSelection } from "./types";

export function profileFor(selection: UnitSelection): UnitProfile | undefined {
  return selection.customProfile ?? unitById(selection.unitId);
}

export function selectionPoints(selection: UnitSelection, list: ArmyList): number {
  const profile = profileFor(selection);
  if (!profile) return 0;

  let pts = 0;
  if (profile.pointsMode === "model") {
    pts = profile.points * selection.models;
  } else {
    const extra = Math.max(0, selection.models - profile.minModels);
    pts = profile.points + extra * (profile.extraModelPoints ?? 0);
  }

  const selected = new Set(selection.optionIds);
  for (const opt of profile.wargearOptions) {
    if (selected.has(opt.id)) pts += opt.points;
  }

  if (selection.enhancementId) {
    const det = detachmentById(list.factionId, list.detachmentId, extrasFor(list));
    const enh = det?.enhancements.find((e) => e.id === selection.enhancementId);
    if (enh) pts += enh.points;
  }

  return pts;
}

export function listPoints(list: ArmyList): number {
  return list.selections.reduce((sum, sel) => sum + selectionPoints(sel, list), 0);
}

export function listValidity(list: ArmyList): { overCap: boolean; empty: boolean } {
  return {
    overCap: listPoints(list) > list.pointsCap,
    empty: list.selections.length === 0,
  };
}
