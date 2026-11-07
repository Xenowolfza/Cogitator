import { detachmentById, extrasFor } from "../data/factions";
import type { ArmyList, ResolvedUnit, UnitSelection, Weapon } from "./types";
import { profileFor, selectionPoints } from "./points";

export function resolveWeapons(selection: UnitSelection): Weapon[] {
  const profile = profileFor(selection);
  if (!profile) return [];

  const selected = new Set(selection.optionIds);
  const options = profile.wargearOptions.filter((o) => selected.has(o.id));
  let weapons = [...profile.weapons];

  for (const opt of options) {
    if (opt.replace) {
      weapons = weapons.filter((w) => w.name !== opt.replace);
    }
    if (opt.remove?.length) {
      const drop = new Set(opt.remove);
      weapons = weapons.filter((w) => !drop.has(w.name));
    }
    if (opt.add?.length) {
      weapons = [...weapons, ...opt.add];
    }
  }

  return weapons;
}

export function resolveSelection(selection: UnitSelection, list: ArmyList): ResolvedUnit | null {
  const profile = profileFor(selection);
  if (!profile) return null;

  const det = detachmentById(list.factionId, list.detachmentId, extrasFor(list));
  const enhancement = selection.enhancementId
    ? det?.enhancements.find((e) => e.id === selection.enhancementId)
    : undefined;

  const selected = new Set(selection.optionIds);
  const invuln = profile.wargearOptions.find((o) => selected.has(o.id) && o.invuln)?.invuln;
  const resolvedProfile = invuln ? { ...profile, stats: { ...profile.stats, invuln } } : profile;

  return {
    selection,
    profile: resolvedProfile,
    weapons: resolveWeapons(selection),
    points: selectionPoints(selection, list),
    enhancement,
  };
}

export function resolveList(list: ArmyList): ResolvedUnit[] {
  return list.selections
    .map((sel) => resolveSelection(sel, list))
    .filter((r): r is ResolvedUnit => r !== null);
}

export function toggleOption(current: string[], optionId: string, group?: string, allGroupIds?: string[]): string[] {
  const has = current.includes(optionId);
  if (group && allGroupIds?.length) {
    const withoutGroup = current.filter((id) => !allGroupIds.includes(id));
    return has ? withoutGroup : [...withoutGroup, optionId];
  }
  return has ? current.filter((id) => id !== optionId) : [...current, optionId];
}
