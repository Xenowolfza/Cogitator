import type { Faction } from "../lib/types";

export const FACTIONS: Faction[] = [
  {
    id: "astartes",
    name: "Adeptus Astartes",
    shortName: "Astartes",
    flavor: "Superhuman infantry, elite characters, and armoured support.",
    detachments: [
      {
        id: "gladius",
        name: "Gladius Task Force",
        summary: "Flexible strike force. Units can shift between assault, tactical, and fire-support postures.",
        enhancements: [
          { id: "ast-blade", name: "Chapter Blade", points: 15, text: "Bearer's melee weapons gain +1 Attack." },
          { id: "ast-halo", name: "Iron Relic", points: 25, text: "Bearer has a 4+ invulnerable save (or improves an existing one by 1, to a maximum of 4+)." },
          { id: "ast-tactical", name: "Tactical Precision", points: 20, text: "Bearer's unit rerolls hit rolls of 1." },
        ],
      },
      {
        id: "firestorm",
        name: "Firestorm Assault",
        summary: "Close-range pressure. Advance and still shoot; transports hit harder after disembarking.",
        enhancements: [
          { id: "fs-flame", name: "Cinder Mantle", points: 20, text: "Bearer's ranged weapons gain [torrent] if they do not already have it, and +1 Strength." },
          { id: "fs-charge", name: "Boarding Fury", points: 15, text: "Bearer and their unit can charge after advancing." },
        ],
      },
    ],
  },
  {
    id: "csm",
    name: "Heretic Astartes",
    shortName: "Chaos Marines",
    flavor: "Traitor legionaries, possessed shock troops, and daemon engines.",
    detachments: [
      {
        id: "slaves",
        name: "Slaves to Darkness",
        summary: "Mark-driven veterans. Dark pacts trade mortal wounds for extra lethality.",
        enhancements: [
          { id: "csm-pact", name: "Dark Boon", points: 20, text: "Once per battle, bearer rerolls all hit and wound rolls in one fight phase." },
          { id: "csm-plate", name: "Warp-forged Plate", points: 15, text: "Bearer has a 4+ invulnerable save." },
          { id: "csm-voice", name: "Voice of the Gods", points: 15, text: "Bearer’s unit is Battle-shocked on a 3+ instead of 2+ when taking a test (they pass more easily)." },
        ],
      },
      {
        id: "pactbound",
        name: "Pactbound Zealots",
        summary: "Devoted killers. Dark pacts become more reliable around the warlord.",
        enhancements: [
          { id: "pz-zeal", name: "Zealot's Brand", points: 15, text: "Bearer’s melee weapons have [lethal hits]." },
        ],
      },
    ],
  },
  {
    id: "necrons",
    name: "Necrons",
    shortName: "Necrons",
    flavor: "Reanimating infantry, destroyer cults, and arcane heavy weapons.",
    detachments: [
      {
        id: "awakened",
        name: "Awakened Dynasty",
        summary: "Noble command. Leader-attached units reanimate more reliably.",
        enhancements: [
          { id: "nec-staff", name: "Void Staff", points: 20, text: "Bearer’s ranged weapons gain +1 Strength and [lethal hits]." },
          { id: "nec-phase", name: "Phase Shifter", points: 15, text: "Bearer has a 4+ invulnerable save." },
          { id: "nec-protocol", name: "Eternal Protocol", points: 25, text: "Bearer’s unit adds 1 to reanimation rolls." },
        ],
      },
      {
        id: "starshatter",
        name: "Starshatter Arsenal",
        summary: "Armour and aircraft. Vehicles gain extra movement and firepower.",
        enhancements: [
          { id: "ss-bequest", name: "Overlord's Bequest", points: 20, text: "Bearer’s unit can shoot after advancing." },
        ],
      },
    ],
  },
  {
    id: "tyranids",
    name: "Tyranids",
    shortName: "Tyranids",
    flavor: "Swarm organisms, synapse control, and living artillery.",
    detachments: [
      {
        id: "invasion",
        name: "Invasion Fleet",
        summary: "Adaptive organisms. Once per battle round, a unit can shift its combat adaptation.",
        enhancements: [
          { id: "ty-synapse", name: "Hive Link", points: 20, text: "Bearer has the Synapse keyword if it did not already, and a 6\" synapse range." },
          { id: "ty-chitin", name: "Hardened Chitin", points: 15, text: "Bearer has a 4+ invulnerable save." },
          { id: "ty-predator", name: "Apex Predator", points: 15, text: "Bearer’s melee weapons gain [lethal hits]." },
        ],
      },
      {
        id: "vanguard",
        name: "Vanguard Onslaught",
        summary: "First-wave hunters. Forward units move and charge more aggressively.",
        enhancements: [
          { id: "vg-lurk", name: "Lurk Instinct", points: 15, text: "Bearer’s unit can charge in the same turn it advanced." },
        ],
      },
    ],
  },
  {
    id: "orks",
    name: "Orks",
    shortName: "Orks",
    flavor: "Mob infantry, smashy characters, and ramshackle vehicles.",
    detachments: [
      {
        id: "warhorde",
        name: "War Horde",
        summary: "Green tide melee. Mob units hit harder on the charge.",
        enhancements: [
          { id: "ork-klaw", name: "Big Boss Klaw", points: 20, text: "Bearer’s melee weapons gain +1 Strength and +1 Damage." },
          { id: "ork-boss", name: "Follow Me, Ladz", points: 15, text: "Bearer’s unit can reroll charge rolls." },
          { id: "ork-iron", name: "Iron Gob", points: 10, text: "Bearer has Feel No Pain 5+." },
        ],
      },
      {
        id: "kult",
        name: "Kult of Speed",
        summary: "Bikes and buggies. Fast units shoot and charge after advancing.",
        enhancements: [
          { id: "ks-go", name: "More Dakka Mount", points: 15, text: "Bearer’s ranged weapons gain [assault] and +1 Attack." },
        ],
      },
    ],
  },
  {
    id: "guard",
    name: "Astra Militarum",
    shortName: "Astra Militarum",
    flavor: "Massed infantry, orders, and super-heavy armour.",
    detachments: [
      {
        id: "combined",
        name: "Combined Regiment",
        summary: "Classic combined arms. Officers issue orders to nearby infantry and tanks.",
        enhancements: [
          { id: "am-laurel", name: "Death Mask", points: 15, text: "Bearer’s unit can reroll Battle-shock tests." },
          { id: "am-blade", name: "Heirloom Blade", points: 10, text: "Bearer’s melee weapons gain +1 Attack and [lethal hits]." },
          { id: "am-scope", name: "Master Vox", points: 20, text: "Bearer can issue an extra Order each turn." },
        ],
      },
      {
        id: "hammer",
        name: "Hammer of the Emperor",
        summary: "Armoured spearhead. Vehicles gain extra movement and rerolls.",
        enhancements: [
          { id: "he-hunter", name: "Hunter's Honour", points: 20, text: "Bearer's vehicle weapons gain [lethal hits] against Monster and Vehicle targets." },
        ],
      },
    ],
  },
];

export function factionById(id: string): Faction | undefined {
  return FACTIONS.find((f) => f.id === id);
}

export function detachmentById(factionId: string, detachmentId: string) {
  const faction = factionById(factionId);
  return faction?.detachments.find((d) => d.id === detachmentId) ?? faction?.detachments[0];
}
