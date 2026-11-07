import type { UnitProfile } from "../lib/types";
import { ASTARTES_UNITS } from "./astartes";
import { CSM_UNITS } from "./csm";
import { GUARD_UNITS } from "./guard";
import { NECRON_UNITS } from "./necrons";
import { ORK_UNITS } from "./orks";
import { TYRANID_UNITS } from "./tyranids";

export const ALL_UNITS: UnitProfile[] = [
  ...ASTARTES_UNITS,
  ...CSM_UNITS,
  ...NECRON_UNITS,
  ...TYRANID_UNITS,
  ...ORK_UNITS,
  ...GUARD_UNITS,
];

export function unitsForFaction(factionId: string): UnitProfile[] {
  return ALL_UNITS.filter((u) => u.factionId === factionId);
}

export function unitById(id: string): UnitProfile | undefined {
  return ALL_UNITS.find((u) => u.id === id);
}
