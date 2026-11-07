import type { UnitProfile } from "../lib/types";
import { factionById } from "./factions";
import { ASTARTES_UNITS } from "./astartes";
import { CSM_UNITS } from "./csm";
import { GUARD_UNITS } from "./guard";
import { NECRON_UNITS } from "./necrons";
import { ORK_UNITS } from "./orks";
import { TYRANID_UNITS } from "./tyranids";
import { BLOOD_ANGELS_UNITS } from "./blood-angels";
import { DARK_ANGELS_UNITS } from "./dark-angels";
import { SPACE_WOLVES_UNITS } from "./space-wolves";
import { BLACK_TEMPLARS_UNITS } from "./black-templars";
import { DEATHWATCH_UNITS } from "./deathwatch";
import { GREY_KNIGHTS_UNITS } from "./grey-knights";
import { CUSTODES_UNITS } from "./custodes";
import { SISTERS_UNITS } from "./sisters";
import { ADMECH_UNITS } from "./admech";
import { KNIGHTS_UNITS } from "./knights";
import { DEATH_GUARD_UNITS } from "./death-guard";
import { THOUSAND_SONS_UNITS } from "./thousand-sons";
import { WORLD_EATERS_UNITS } from "./world-eaters";
import { DAEMONS_UNITS } from "./daemons";
import { CHAOS_KNIGHTS_UNITS } from "./chaos-knights";
import { TAU_UNITS } from "./tau";
import { AELDARI_UNITS } from "./aeldari";
import { DRUKHARI_UNITS } from "./drukhari";
import { VOTANN_UNITS } from "./votann";
import { GSC_UNITS } from "./gsc";

export const ALL_UNITS: UnitProfile[] = [
  ...ASTARTES_UNITS,
  ...CSM_UNITS,
  ...NECRON_UNITS,
  ...TYRANID_UNITS,
  ...ORK_UNITS,
  ...GUARD_UNITS,
  ...BLOOD_ANGELS_UNITS,
  ...DARK_ANGELS_UNITS,
  ...SPACE_WOLVES_UNITS,
  ...BLACK_TEMPLARS_UNITS,
  ...DEATHWATCH_UNITS,
  ...GREY_KNIGHTS_UNITS,
  ...CUSTODES_UNITS,
  ...SISTERS_UNITS,
  ...ADMECH_UNITS,
  ...KNIGHTS_UNITS,
  ...DEATH_GUARD_UNITS,
  ...THOUSAND_SONS_UNITS,
  ...WORLD_EATERS_UNITS,
  ...DAEMONS_UNITS,
  ...CHAOS_KNIGHTS_UNITS,
  ...TAU_UNITS,
  ...AELDARI_UNITS,
  ...DRUKHARI_UNITS,
  ...VOTANN_UNITS,
  ...GSC_UNITS,
];

export function unitsForFaction(factionId: string): UnitProfile[] {
  const faction = factionById(factionId);
  const ids = new Set([factionId, ...(faction?.includeFactionIds ?? [])]);
  return ALL_UNITS.filter((u) => ids.has(u.factionId)).sort((a, b) => {
    if (a.factionId === factionId && b.factionId !== factionId) return -1;
    if (b.factionId === factionId && a.factionId !== factionId) return 1;
    return a.name.localeCompare(b.name);
  });
}

export function unitById(id: string): UnitProfile | undefined {
  return ALL_UNITS.find((u) => u.id === id);
}
