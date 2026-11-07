export type UnitRole = "character" | "battleline" | "dedicatedTransport" | "other";

export type Weapon = {
  name: string;
  kind: "ranged" | "melee";
  range: string;
  attacks: string;
  skill: string;
  strength: string;
  ap: string;
  damage: string;
  keywords: string[];
};

export type Statline = {
  m: string;
  t: string;
  sv: string;
  w: string;
  ld: string;
  oc: string;
  invuln?: string;
};

export type WargearOption = {
  id: string;
  label: string;
  points: number;
  group?: string;
  replace?: string;
  add?: Weapon[];
  remove?: string[];
  invuln?: string;
};

export type Ability = {
  name: string;
  text: string;
};

export type UnitProfile = {
  id: string;
  name: string;
  factionId: string;
  role: UnitRole;
  points: number;
  pointsMode: "unit" | "model";
  extraModelPoints?: number;
  minModels: number;
  maxModels: number;
  stats: Statline;
  keywords: string[];
  factionKeywords: string[];
  coreAbilities: string[];
  factionAbility?: string;
  weapons: Weapon[];
  wargearOptions: WargearOption[];
  abilities: Ability[];
  composition: string;
  damaged?: { threshold: string; profile: string };
};

export type Enhancement = {
  id: string;
  name: string;
  points: number;
  text: string;
};

export type Detachment = {
  id: string;
  name: string;
  summary: string;
  enhancements: Enhancement[];
  /** User-added note. Catalog detachments omit this. */
  custom?: boolean;
  /** Filename of a private reference stored only in this browser. */
  referenceName?: string;
  referenceId?: string;
};

export type Allegiance = "imperium" | "chaos" | "xenos";

export type Faction = {
  id: string;
  name: string;
  shortName: string;
  flavor: string;
  allegiance: Allegiance;
  includeFactionIds?: string[];
  detachments: Detachment[];
};

export type UnitSelection = {
  uid: string;
  unitId: string;
  models: number;
  optionIds: string[];
  enhancementId?: string;
  customProfile?: UnitProfile;
};

export type FactionStashEntry = {
  detachmentId: string;
  selections: UnitSelection[];
};

export type CardThemeId =
  | "parchment"
  | "blood"
  | "waaagh"
  | "ultramarine"
  | "gold"
  | "night"
  | "necrodermis"
  | "ivory"
  | "purple"
  | "fenris";

export type ArmyList = {
  id: string;
  name: string;
  factionId: string;
  detachmentId: string;
  pointsCap: number;
  cardTheme?: CardThemeId;
  selections: UnitSelection[];
  /** User-written detachment notes, keyed by faction id. */
  extraDetachments?: Record<string, Detachment[]>;
  factionStash?: Record<string, FactionStashEntry>;
  updatedAt: string;
};

export type AppStore = {
  lists: ArmyList[];
  activeId: string;
};

export type ResolvedUnit = {
  selection: UnitSelection;
  profile: UnitProfile;
  weapons: Weapon[];
  points: number;
  enhancement?: Enhancement;
};
