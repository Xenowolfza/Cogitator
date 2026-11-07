import type { UnitProfile, Weapon } from "../lib/types";

export function ranged(
  name: string,
  range: string,
  attacks: string,
  skill: string,
  strength: string,
  ap: string,
  damage: string,
  keywords: string[] = [],
): Weapon {
  return { name, kind: "ranged", range, attacks, skill, strength, ap, damage, keywords };
}

export function melee(
  name: string,
  attacks: string,
  skill: string,
  strength: string,
  ap: string,
  damage: string,
  keywords: string[] = [],
): Weapon {
  return { name, kind: "melee", range: "Melee", attacks, skill, strength, ap, damage, keywords };
}

export function unit(profile: UnitProfile): UnitProfile {
  return profile;
}
