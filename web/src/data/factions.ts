import type { Faction } from "../lib/types";

export const FACTIONS: Faction[] = [
  {
    id: "astartes",
    name: "Adeptus Astartes",
    shortName: "Astartes",
    flavor: "Superhuman infantry, elite characters, and armoured support.",
    allegiance: "imperium",
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
    allegiance: "chaos",
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
    allegiance: "xenos",
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
    allegiance: "xenos",
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
    allegiance: "xenos",
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
    allegiance: "imperium",
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
  {
    id: "blood-angels",
    name: "Blood Angels",
    shortName: "Blood Angels",
    flavor: "Assault specialists with Death Company and Sanguinary Guard.",
    allegiance: "imperium",
    includeFactionIds: ["astartes"],
    detachments: [
      {
        id: "sons-sanguinius",
        name: "Sons of Sanguinius",
        summary: "Jump-pack assault. Charge bonuses and extra attacks on the turn they charge.",
        enhancements: [
          { id: "ba-rage", name: "Red Thirst", points: 15, text: "Bearer's melee weapons gain +1 Strength on the charge." },
          { id: "ba-wings", name: "Angel's Pinions", points: 20, text: "Bearer and their unit can charge after advancing." },
        ],
      },
    ],
  },
  {
    id: "dark-angels",
    name: "Dark Angels",
    shortName: "Dark Angels",
    flavor: "Inner Circle veterans, Deathwing terminators, and Ravenwing outriders.",
    allegiance: "imperium",
    includeFactionIds: ["astartes"],
    detachments: [
      {
        id: "unforgiven",
        name: "Unforgiven Task Force",
        summary: "Deathwing and Ravenwing support. Inner Circle units reroll hits.",
        enhancements: [
          { id: "da-sword", name: "Heavenfall Blade", points: 20, text: "Bearer's melee weapons gain [devastating wounds]." },
        ],
      },
    ],
  },
  {
    id: "space-wolves",
    name: "Space Wolves",
    shortName: "Space Wolves",
    flavor: "Pack hunters, thunderwolf cavalry, and saga-driven characters.",
    allegiance: "imperium",
    includeFactionIds: ["astartes"],
    detachments: [
      {
        id: "champions-fenris",
        name: "Champions of Fenris",
        summary: "Pack tactics. Infantry gain extra movement and charge rerolls.",
        enhancements: [
          { id: "sw-saga", name: "Saga of the Bear", points: 15, text: "Bearer has Feel No Pain 5+." },
        ],
      },
    ],
  },
  {
    id: "black-templars",
    name: "Black Templars",
    shortName: "Black Templars",
    flavor: "Zealous crusaders, sword brethren, and vow-driven melee.",
    allegiance: "imperium",
    includeFactionIds: ["astartes"],
    detachments: [
      {
        id: "righteous-crusaders",
        name: "Righteous Crusaders",
        summary: "Take a vow. Units fight on death or gain lethal hits.",
        enhancements: [
          { id: "bt-vow", name: "Witchseeker Vow", points: 15, text: "Bearer's weapons have [anti-psyker 4+] and [precision]." },
        ],
      },
    ],
  },
  {
    id: "deathwatch",
    name: "Deathwatch",
    shortName: "Deathwatch",
    flavor: "Kill teams with mixed loadouts and xenophase weaponry.",
    allegiance: "imperium",
    includeFactionIds: ["astartes"],
    detachments: [
      {
        id: "black-spear",
        name: "Black Spear Task Force",
        summary: "Mission tactics. Kill teams can switch special-issue ammunition.",
        enhancements: [
          { id: "dw-ammo", name: "Special Issue Cache", points: 15, text: "Bearer's ranged weapons gain [lethal hits] against Xenos." },
        ],
      },
    ],
  },
  {
    id: "grey-knights",
    name: "Grey Knights",
    shortName: "Grey Knights",
    flavor: "Daemon-hunting psykers in terminator plate.",
    allegiance: "imperium",
    detachments: [
      {
        id: "teleport-strike",
        name: "Teleport Strike Force",
        summary: "Deep-striking psykers. Units can teleport in the movement phase.",
        enhancements: [
          { id: "gk-ward", name: "Aegis Ward", points: 20, text: "Bearer has a 4+ invulnerable save and Feel No Pain 5+ against Psychic attacks." },
        ],
      },
    ],
  },
  {
    id: "custodes",
    name: "Adeptus Custodes",
    shortName: "Custodes",
    flavor: "Golden demigods, unyielding infantry, and dreadnought support.",
    allegiance: "imperium",
    detachments: [
      {
        id: "shield-host",
        name: "Shield Host",
        summary: "Martial ka'tah. Switch between offensive and defensive stances.",
        enhancements: [
          { id: "ac-auramite", name: "Auramite Plate", points: 25, text: "Improve the bearer's Save by 1, to a maximum of 2+." },
        ],
      },
    ],
  },
  {
    id: "sisters",
    name: "Adepta Sororitas",
    shortName: "Sisters",
    flavor: "Faithful infantry, miracle dice, and sanctified war engines.",
    allegiance: "imperium",
    detachments: [
      {
        id: "triumph-faith",
        name: "Triumph of Saint Katherine",
        summary: "Acts of Faith. Miracle dice fuel extra hits and saves.",
        enhancements: [
          { id: "as-relic", name: "Martyr's Vestment", points: 15, text: "Bearer has a 4+ invulnerable save and Feel No Pain 6+." },
        ],
      },
    ],
  },
  {
    id: "admech",
    name: "Adeptus Mechanicus",
    shortName: "Adeptus Mechanicus",
    flavor: "Skitarii cohorts, holy engines, and rad-saturated firepower.",
    allegiance: "imperium",
    detachments: [
      {
        id: "rad-zone",
        name: "Rad-Zone Corps",
        summary: "Doctrina imperatives. Toggle hit and save modifiers each battle round.",
        enhancements: [
          { id: "am-canticles", name: "Canticle Boost", points: 15, text: "Bearer's unit rerolls hit rolls of 1." },
        ],
      },
    ],
  },
  {
    id: "knights",
    name: "Imperial Knights",
    shortName: "Imperial Knights",
    flavor: "Super-heavy walkers and Armiger support.",
    allegiance: "imperium",
    detachments: [
      {
        id: "noble-lance",
        name: "Noble Lance",
        summary: "Bondsman abilities. Big knights buff nearby Armigers.",
        enhancements: [
          { id: "ik-heirlooms", name: "Heedless Charge", points: 20, text: "Bearer can charge after advancing." },
        ],
      },
    ],
  },
  {
    id: "death-guard",
    name: "Death Guard",
    shortName: "Death Guard",
    flavor: "Plague infantry, contagions, and foetid daemon engines.",
    allegiance: "chaos",
    detachments: [
      {
        id: "virulent-vectorium",
        name: "Virulent Vectorium",
        summary: "Contagion range. Enemy toughness drops near your units.",
        enhancements: [
          { id: "dg-rot", name: "Rot-hive Plate", points: 20, text: "Bearer has Feel No Pain 5+ and a 4+ invulnerable save." },
        ],
      },
    ],
  },
  {
    id: "thousand-sons",
    name: "Thousand Sons",
    shortName: "Thousand Sons",
    flavor: "Sorcerers, Rubric infantry, and cabalistic rituals.",
    allegiance: "chaos",
    detachments: [
      {
        id: "cult-of-magic",
        name: "Cult of Magic",
        summary: "Cabal of sorcerers. Spend cabal points for extra psychic power.",
        enhancements: [
          { id: "ts-arcana", name: "Arcane Focus", points: 20, text: "Bearer's Psychic weapons gain +1 Strength and [devastating wounds]." },
        ],
      },
    ],
  },
  {
    id: "world-eaters",
    name: "World Eaters",
    shortName: "World Eaters",
    flavor: "Berzerker infantry, blessings of Khorne, and daemon engines.",
    allegiance: "chaos",
    detachments: [
      {
        id: "berzerker-warband",
        name: "Berzerker Warband",
        summary: "Blessings of Khorne. Extra attacks and advance-and-charge.",
        enhancements: [
          { id: "we-axe", name: "Gorefather", points: 20, text: "Bearer's melee weapons gain +1 Damage and [lethal hits]." },
        ],
      },
    ],
  },
  {
    id: "daemons",
    name: "Chaos Daemons",
    shortName: "Daemons",
    flavor: "Warp-spawned infantry, greater daemons, and soul grinders.",
    allegiance: "chaos",
    detachments: [
      {
        id: "daemonic-incursion",
        name: "Daemonic Incursion",
        summary: "Shadow of Chaos. Units in your shadow gain extra durability.",
        enhancements: [
          { id: "cd-crown", name: "Crown of the Pantheon", points: 25, text: "Bearer has a 4+ invulnerable save and Feel No Pain 5+." },
        ],
      },
    ],
  },
  {
    id: "chaos-knights",
    name: "Chaos Knights",
    shortName: "Chaos Knights",
    flavor: "Fallen walkers and War Dog packs.",
    allegiance: "chaos",
    detachments: [
      {
        id: "traitor-isab",
        name: "Traitoris Lance",
        summary: "Dread abilities. Super-heavies aura-buff War Dogs.",
        enhancements: [
          { id: "ck-warp", name: "Warp-borne Hatred", points: 20, text: "Bearer's weapons gain [lethal hits]." },
        ],
      },
    ],
  },
  {
    id: "tau",
    name: "T'au Empire",
    shortName: "T'au",
    flavor: "Crisis suits, markerlights, and long-range cadre firepower.",
    allegiance: "xenos",
    detachments: [
      {
        id: "kauyon",
        name: "Kauyon",
        summary: "Patient hunter. Guided units gain extra AP later in the battle.",
        enhancements: [
          { id: "tau-ml", name: "Prototype Uplink", points: 15, text: "Bearer's ranged weapons have [ignores cover] and [lethal hits]." },
        ],
      },
    ],
  },
  {
    id: "aeldari",
    name: "Aeldari",
    shortName: "Aeldari",
    flavor: "Aspect warriors, seers, and skimming tanks.",
    allegiance: "xenos",
    detachments: [
      {
        id: "warhost",
        name: "Warhost",
        summary: "Strands of Fate. Reroll dice and fade after shooting.",
        enhancements: [
          { id: "ae-fate", name: "Mantle of Fates", points: 20, text: "Once per turn, bearer can reroll one hit, wound, or save." },
        ],
      },
    ],
  },
  {
    id: "drukhari",
    name: "Drukhari",
    shortName: "Drukhari",
    flavor: "Raiders, wracks, and pain-fuelled glass cannons.",
    allegiance: "xenos",
    detachments: [
      {
        id: "realspace-raiders",
        name: "Realspace Raiders",
        summary: "Power from Pain. As they kill, they get faster and harder to hurt.",
        enhancements: [
          { id: "dk-pain", name: "Crucible of Malediction", points: 15, text: "Bearer’s melee weapons have [anti-infantry 3+] and [lance]." },
        ],
      },
    ],
  },
  {
    id: "votann",
    name: "Leagues of Votann",
    shortName: "Votann",
    flavor: "Kin infantry, sagitaur transports, and hearthguard elites.",
    allegiance: "xenos",
    detachments: [
      {
        id: "oathband",
        name: "Oathband",
        summary: "Judgement tokens. Marked enemies are easier to wound.",
        enhancements: [
          { id: "lov-beam", name: "Ancestral Crest", points: 15, text: "Bearer's ranged weapons gain [lethal hits] and [heavy]." },
        ],
      },
    ],
  },
  {
    id: "gsc",
    name: "Genestealer Cults",
    shortName: "Genestealer Cults",
    flavor: "Ambush infantry, hybrid aberrants, and stolen industrial vehicles.",
    allegiance: "xenos",
    detachments: [
      {
        id: "host-of-asc",
        name: "Host of Ascension",
        summary: "Cult ambush. Units return from reserves more aggressively.",
        enhancements: [
          { id: "gsc-focus", name: "Patriarch's Gift", points: 15, text: "Bearer’s unit can charge after advancing." },
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
