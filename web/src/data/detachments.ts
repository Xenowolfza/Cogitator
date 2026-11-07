import type { Detachment } from "../lib/types";

function pack(rows: [string, string][]): Detachment[] {
  return rows.map(([name, summary]) => ({
    id: name
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-|-$/g, ""),
    name,
    summary,
    enhancements: [],
  }));
}

/** Playstyle notes only. Official wording stays in your books and faction packs. */
export const ASTARTES_DETS = pack([
  ["Gladius Task Force", "Flexible company that can lean into shooting, melee, or a mixed plan with almost any unit."],
  ["Anvil Siege Force", "Dug-in firepower. Heavy weapons and units that want to stay still and shoot."],
  ["Firestorm Assault Force", "Close-range heat. Favours torrent weapons and aggressive short-range shooting."],
  ["Stormlance Task Force", "Fast assault. Mounted units and a few vehicles that want to move and charge."],
  ["Vanguard Spearhead", "Recon and ambush. Phobos and scout elements that strike and fade."],
  ["Ironstorm Spearhead", "Armour spearhead. Vehicles and walkers built to trade shots and hold ground."],
  ["1st Company Task Force", "Veteran company. Terminators and veteran infantry as the main striking arm."],
  ["Librarius Conclave", "Psychic cadre. Librarians and other psykers shaping the battle from the back line."],
  ["Armoured Speartip", "Heavy transport push. Built around big transports delivering a punch."],
  ["Bastion Task Force", "Objective holders. Battleline infantry that want to sit on ground and endure."],
  ["Orbital Assault Force", "Drop assault. Forces that arrive by drop pod and fight where they land."],
  ["Fulguris Task Force", "Speeder screen. Land speeders used to harass and pick targets."],
  ["Subversion Assets", "Infiltration. Scout squads and Phobos units that start forward."],
  ["Headhunter Task Force", "Tank hunting. A gunline aimed at priority enemy armour."],
  ["Ceramite Sentinels", "Terrain infantry. Foot units that fight better when they use cover."],
  ["Blade of Ultramar", "Ultramarines-style all-rounders. A broad force rather than a single specialist wing."],
  ["Emperor's Shield", "Imperial Fists flavour. Terminators and veterans holding a chosen point."],
  ["Forgefather's Seekers", "Salamanders flavour. Short-range guns and close firefights."],
  ["Hammer of Avernii", "Iron Hands flavour. Terminators, veterans, and dreadnoughts."],
  ["Reclamation Force", "Ultramarines melee. Infantry that want to take ground in close combat."],
  ["Shadowmark Talon", "Raven Guard flavour. Phobos and scouts that disrupt rather than grind."],
  ["Spearpoint Task Force", "White Scars flavour. Mounted units, speeders, and aircraft."],
]);

export const BLOOD_ANGELS_DETS = pack([
  ["Liberator Assault Group", "Broad assault host. The whole army is pointed at a charge."],
  ["The Angelic Host", "Jump-pack wing. Fast infantry that want to arrive and strike."],
  ["Angelic Inheritors", "Character-led assault. Heroes carry the plan more than the rank and file."],
  ["The Lost Brethren", "Death Company focus. Furious melee units that are hard to keep back."],
  ["Rage-cursed Onslaught", "Wider Death Company push. The doomed assault is the whole army, not a side unit."],
  ["Encarmine Speartip", "Sanguinary Guard spear. A small elite that hunts one target."],
  ["Legacy of Grace", "Heroic bloodline. Characters are the reason the list exists."],
  ["Wrath of the Doomed", "Death Company strike. A narrow list built on the black-armoured assault."],
]);

export const DARK_ANGELS_DETS = pack([
  ["Unforgiven Task Force", "Mixed chapter. Works with the whole roster rather than one wing."],
  ["Inner Circle Task Force", "Deathwing. Terminators and the Inner Circle as the core."],
  ["Company of Hunters", "Ravenwing. Mounted hunters that want speed and charges."],
  ["Lion's Blade Task Force", "Both wings. A balance of Ravenwing speed and Deathwing staying power."],
  ["Wrath of the Rock", "Foot-slogging wings. Deathwing and Ravenwing that are not built around vehicles."],
  ["Dark Age Arsenal", "Plasma arsenal. A small force built around plasma weapons."],
  ["Darkflight Pursuit", "Flying Ravenwing. The part of the wing that can actually fly."],
  ["Interrogation Conclave", "Chaplains. A narrow force led by the Interrogator-Chaplains."],
]);

export const SPACE_WOLVES_DETS = pack([
  ["Saga of the Beastslayer", "Broad hunt. The whole pack is pointed at big targets."],
  ["Saga of the Bold", "Heroes of the rout. Characters carry the saga."],
  ["Saga of the Hunter", "Grey Hunters and Blood Claws. The classic pack hunt."],
  ["Saga of the Great Wolf", "Wolf Guard around Logan's legend. Elite infantry and the Great Wolf's household."],
  ["Champions of Fenris", "A small band of characters rather than a wide army."],
  ["Legends of Saga and Song", "Terminator sagas. A narrow list of heavy infantry."],
  ["Veterans of the Fang", "Grey Hunter veterans. A focused pack, not the whole chapter."],
]);

export const BLACK_TEMPLARS_DETS = pack([
  ["Companions of Vehemence", "Crusade infantry. Foot units that want to close and fight."],
  ["Godhammer Assault Force", "Transport assault. The crusade rides in and hits on the disembark."],
  ["Vindication Task Force", "Crusader squads and Ancients. A vow built around the battleline and standards."],
  ["Marshal's Household", "Sword Brethren. A small household of the marshal's elites."],
  ["The Living Miracle", "The Emperor's Champion. A narrow list built around that duel."],
  ["Wrathful Procession", "Infantry and mounted crusaders advancing as a procession."],
]);

export const DEATHWATCH_DETS = pack([
  ["Black Spear Task Force", "The kill-team host. Mixed specialists and mission tactics for the whole watch."],
]);

export const GREY_KNIGHTS_DETS = pack([
  ["Warpbane Task Force", "Purifiers at the centre. The army is built to burn daemons and psykers."],
  ["Banishers", "Melee brotherhood. The whole force wants to fight up close."],
  ["Brotherhood Strike", "Balanced strike. A general teleport assault rather than one squad type."],
  ["Hallowed Conclave", "Terminator host. The heavy plate is the core of the list."],
  ["Sanctic Spearhead", "Dreadknights and vehicles. The big suits do the hunting."],
  ["Augurium Task Force", "Scouting strike. A broader force that wants information and angles."],
  ["Argent Assault", "Paladin assault. A narrow list of the heaviest infantry."],
  ["Fires of Purgation", "Purgation squads. A narrow shooting wing."],
  ["Immaterial Interdiction", "Interceptors. A narrow list built on personal teleporters."],
]);

export const CUSTODES_DETS = pack([
  ["Shield Host", "Classic host. The whole golden line, not one specialist wing."],
  ["Lions of the Emperor", "Terminators and characters. Disruption through elite blades."],
  ["Auric Champions", "Heroes. A smaller list where each character is the plan."],
  ["Solar Spearhead", "Vehicles and walkers. The spear is the armour, not the infantry."],
  ["Talons of the Emperor", "Custodes and the Silent Sisterhood together. Needs both halves."],
  ["Null Maiden Vigil", "Anathema Psykana. A sisterhood-led force rather than a golden wall."],
  ["Might of the Moritoi", "Walkers. Dreadnoughts as the main threat."],
  ["Silent Hunters", "A small Anathema Psykana hunting party."],
  ["Tharanatoi Hammerblow", "Allarus and other terminators as a single hammer."],
]);

export const SISTERS_DETS = pack([
  ["Hallowed Martyrs", "The whole order. Works when any unit is asked to hold or die well."],
  ["Bringers of Flame", "Short-range fire. Flamers and close guns are the point of the list."],
  ["Army of Faith", "Jump troops and acts of faith. A mobile order rather than a castle."],
  ["Champions of Faith", "Melee champions. Sacresants and fighters who want to disrupt."],
  ["Penitent Host", "Penitent engines and the mortifiers' kin. The sinners are the army."],
  ["Chorus of Condemnation", "A small jump-pack chorus, not the full order."],
  ["Sacred Champions", "Celestian Sacresants as a narrow bodyguard host."],
  ["Sanctified Orators", "Characters. A small list built around the speakers of the faith."],
]);

export const ADMECH_DETS = pack([
  ["Haloscreed Battle Clade", "Broad forge war. The whole army, not one clade."],
  ["Rad-zone Corps", "Doctrina and rad-saturated ground. A general Skitarii war."],
  ["Skitarii Hunter Cohort", "Skitarii hunters. Infantry that want to scout and pick targets."],
  ["Cohort Cybernetica", "Robots and vehicles. Legio Cybernetica as the core."],
  ["Data-psalm Conclave", "Cult Mechanicus. The priests and their congregations, not the rangers."],
  ["Explorator Maniple", "Transports. A force that wants to ride and claim a site."],
  ["Eradication Cohort", "Skitarii eradication. A wide infantry gunline."],
  ["Cohort Acquisitus", "Cavalry, jump troops, rangers, and infiltrators. A small acquisition force."],
  ["Lords of the Forge", "Tech-Priests. A narrow list of the priesthood."],
  ["Luminen Auto-Choir", "Electropriests. A narrow close-range cult."],
]);

export const KNIGHTS_DETS = pack([
  ["Questoris Companions", "A mixed lance that wants to take ground with the big knights and their squires."],
  ["Freeblade Company", "A broad freeblade host. Any knight pattern can carry the plan."],
  ["Valourstrike Lance", "Aggressive lance. The army is pointed at a purge, not a sit-and-shoot."],
  ["Gate Warden Lance", "Ranged wardens. Big guns holding a gate."],
  ["Spearhead-at-arms", "Armigers. The smaller knights do the scouting and the work."],
  ["Dominus Foebreakers", "Dominus-class only. A single huge breaker, not a mixed lance."],
  ["Questor Forgepact", "Knights with Adeptus Mechanicus support. A narrow allied forge pact."],
  ["Throne-bonded Outriders", "A small Armiger outrider screen."],
]);

export const GUARD_DETS = pack([
  ["Combined Arms", "Classic combined regiment. Infantry, tanks, and orders across the whole army."],
  ["Hammer of the Emperor", "Tank company. Armour is the plan."],
  ["Mechanised Assault", "Infantry in transports. The punch comes after they disembark."],
  ["Armoured Infantry", "Tanks, transports, and the squads that ride in them."],
  ["Siege Regiment", "Infantry and artillery. A grinding siege rather than a cavalry charge."],
  ["Recon Element", "Sentinels and light infantry. A wide scouting force."],
  ["Grizzled Company", "Veterans of the line. A broad force that is not tied to one vehicle type."],
  ["Steel Hammer", "Super-heavies and tanks. The biggest guns."],
  ["Bridgehead Strike", "Tempestus. A small airborne or drop bridgehead."],
  ["Designation Force", "Scout sentinels and light infantry marking targets."],
  ["Abhuman Auxiliaries", "Abhumans. A narrow allied contingent, not the whole regiment."],
]);

export const CSM_DETS = pack([
  ["Pactbound Zealots", "Marks and dark pacts across a broad traitor host."],
  ["Renegade Raiders", "Astartes and transports that want to raid and reposition."],
  ["Veterans of the Long War", "Legionary infantry that want to take and hold."],
  ["Creations of Bile", "Augmented infantry, with Fabius Bile as the natural centre."],
  ["Soulforged Warpack", "Daemon engines and possessed vehicles."],
  ["Fellhammer Siege-host", "Ranged traitors. A siege line rather than a charge."],
  ["Deceptors", "Astartes and cultists who want to misdirect."],
  ["Dread Talons", "Fast infantry that hunt isolated targets."],
  ["Chaos Cult", "The Damned. Cultists and the mortal host, not the legionaries."],
  ["Renegade Warband", "A general warband that is not built on the god-specific cult troops."],
  ["Cult of the Arkifane", "Vashtorr's vehicles. A warpsmith's engine park."],
  ["Huron's Marauders", "Red Corsairs. A raider force with a wide brief."],
  ["Nightmare Hunt", "Jump infantry and the Nemesis Claw."],
  ["Warpstrike Champions", "Obliterators, mutilators, and terminators."],
  ["Cabal of Chaos", "A small psyker cabal."],
  ["Devotees of Destruction", "Havocs and obliterators. A small heavy-weapons cult."],
  ["Murdertalon Raiders", "Jump infantry as a single raiding claw."],
]);

export const DEATH_GUARD_DETS = pack([
  ["Virulent Vectorium", "The whole legion. Contagion across every unit."],
  ["Champions of Contagion", "Characters spreading the gift. Heroes are the vectors."],
  ["Mortarion's Hammer", "Vehicles. The plague engines do the killing."],
  ["Death Lord's Chosen", "Terminators. Blightlords and Deathshroud as the core."],
  ["Shamblerot Vectorium", "Poxwalkers. The walking pox is the army."],
  ["Tallyband Summoners", "Daemons of Nurgle fighting beside the legion."],
  ["Flyblown Host", "A small character-led host."],
  ["Contagion Engines", "Plague Marines as a narrow vector."],
  ["Paragons of Putrescence", "Bloat-drones, blight-haulers, and helbrutes."],
]);

export const THOUSAND_SONS_DETS = pack([
  ["Grand Coven", "The whole cult. Sorcerers and rubricae together."],
  ["Rubricae Phalanx", "Rubric Marines and Scarab Occult as a durable phalanx."],
  ["Hexwarp Thrallband", "A broader thrallband that is not locked to one unit type."],
  ["Warpforged Cabal", "Vehicles. The silver towers and predators carry the plan."],
  ["Warpmeld Pact", "Tzaangors, spawn, and the Mutalith. The mutated host."],
  ["Changehost of Deceit", "Daemons of Tzeentch beside the legion. A scouting deceit."],
  ["Ritual of Regeneration", "A small psyker ritual, not the full coven."],
  ["Sekhetar Cohort", "Sekhetar robots. A narrow automaton cohort."],
  ["Servants of Change", "Tzaangors and mutants as a small screening force."],
]);

export const WORLD_EATERS_DETS = pack([
  ["Berzerker Warband", "The whole warband. Blessings and a general assault."],
  ["Goretrack Onslaught", "Transports. The berzerkers ride in and spill out."],
  ["Khorne Daemonkin", "Daemons beside the World Eaters. A mixed recon slaughter."],
  ["Possessed Slaughterband", "Eightbound and daemon princes. The possessed are the army."],
  ["Cult of Blood", "Jakhal mortals, monsters, and the biggest of Khorne's engines."],
  ["Butchers of Khorne", "Terminators. A narrow butcher's bloc."],
  ["Brazen Engines", "Daemon engines. A small engine pack."],
  ["Vessels of Wrath", "Characters who are not epic heroes. A narrow champion list."],
]);

export const DAEMONS_DETS = pack([
  ["Daemonic Incursion", "The whole pantheon. A mixed incursion rather than one god."],
  ["Blood Legion", "Khorne. Bloodletters, hounds, and the bloodthirsters' kin."],
  ["Plague Legion", "Nurgle. Slow, tough daemons that want to hold ground."],
  ["Scintillating Legion", "Tzeentch. Horrors and the glittering host."],
  ["Legion of Excess", "Slaanesh. Daemonettes and the excess host."],
  ["Shadow Legion", "Be'lakor's shadow, including heretic astartes that fight in it."],
  ["Cavalcade of Chaos", "Mounted daemons. A small cavalry cavalcade."],
  ["Lords of the Warp", "Lesser characters, not the greater daemons."],
  ["Warptide", "Battleline daemons as a screening tide."],
]);

export const CHAOS_KNIGHTS_DETS = pack([
  ["Infernal Lance", "A broad fallen lance. Any pattern can carry a purge."],
  ["Traitoris Lance", "A general traitor lance, big knights and War Dogs together."],
  ["Lords of Dread", "Dread auras across the lance. The big knights set the tone."],
  ["Houndpack Lance", "War Dogs. The pack hunts without a single dominant tyrant."],
  ["Helhunt Lance", "A big knight paired with War Dogs."],
  ["Bastions of Tyranny", "Knight Tyrants. A single dominus-class bastion."],
  ["Hunting Warpack", "A small War Dog hunting pack."],
  ["Iconoclast Fiefdom", "Knights with damned infantry. A narrow household."],
]);

export const NECRON_DETS = pack([
  ["Awakened Dynasty", "Characters leading their units. The classic noble command."],
  ["Starshatter Arsenal", "Vehicles and mounted tombs. The arsenal does the shooting."],
  ["Canoptek Court", "Canoptek constructs and crypteks."],
  ["Hypercrypt Legion", "Monoliths and sudden relocation. A repositioning legion."],
  ["Obeisance Phalanx", "Overlords, lychguard, and the Triarch."],
  ["Annihilation Legion", "Destroyer Cult and flayed ones. A hunting legion."],
  ["Cursed Legion", "Destroyers. A narrower destroyer host."],
  ["Cryptek Conclave", "Crypteks. The technomancers are the core."],
  ["Pantheon of Woe", "C'tan. The star gods are the plan."],
  ["Hand of the Dynasty", "Immortals and warriors. A small dynastic fist."],
  ["Skyshroud Spearhead", "Mounted units. A small flying or riding spearhead."],
  ["The Phaeron's Armoury", "Monoliths and tesseract vaults. A single armoury piece."],
]);

export const TYRANID_DETS = pack([
  ["Invasion Fleet", "The general swarm. Any organism can carry an adaptation."],
  ["Vanguard Onslaught", "Lictors, leapers, and the first wave."],
  ["Synaptic Nexus", "Synapse creatures. The hive mind's nodes are the army."],
  ["Unending Swarm", "Endless multitude. Gaunts and the bodies that replace themselves."],
  ["Assimilation Swarm", "Harvesters. Units that want to feed and recover."],
  ["Crusher Stampede", "Monsters. The big organisms do the crushing."],
  ["Subterranean Assault", "Burrowers. Units that arrive from below."],
  ["Ambush Predators", "Lictors. A small ambush, not the whole hive."],
  ["Warrior Bioform Onslaught", "Warriors. A narrow warrior host."],
  ["Talons of the Norn Queen", "Norn emissaries and assimilators. A single huge talon."],
]);

export const ORK_DETS = pack([
  ["War Horde", "The general Waaagh. Any mob can carry a straight fight."],
  ["Green Tide", "Boyz. A wide tide of infantry that wants to hold ground."],
  ["Bully Boyz", "Warboss, nobz, and meganobz. The big lads do the bullying."],
  ["Kult of Speed", "Speed Freeks. Bikes and buggies that want to disrupt."],
  ["Speedwaaagh!", "Speed Freeks on a wider hunt. Scouting as much as smashing."],
  ["Da Big Hunt", "Beast Snaggas. Built to drag down monsters and vehicles."],
  ["Dread Mob", "Walkers. Deff dreads and the killa kans' cousins."],
  ["Blitz Brigade", "Transports. The boyz ride in and spill out."],
  ["Freebooter Krew", "Flash Gitz and a mix of infantry, mounted, and walkers."],
  ["More Dakka!", "Ranged infantry. A small mob that wants to shoot."],
  ["Rollin' Deff", "Battlewagons, hunta rigs, and kill rigs."],
  ["Taktikal Brigade", "Stormboyz, boyz, and kommandos. A small planned raid."],
]);

export const TAU_DETS = pack([
  ["Mont'ka", "Mobile killing blow. The army wants to strike early and move."],
  ["Kauyon", "Patient hunter. Static guns that want a later, heavier volley."],
  ["Retaliation Cadre", "Battlesuits. Crisis and the bigger suits are the army."],
  ["Kroot Hunting Pack", "Kroot. The auxiliaries are the main force, not a screen."],
  ["Advanced Acquisition Cadre", "Pathfinders and stealth suits. A small markerlight cadre."],
  ["Auxiliary Cadre", "Kroot, Vespid, and stealth suits together."],
  ["Experimental Prototype Cadre", "Battlesuit characters. A small prototype team."],
]);

export const AELDARI_DETS = pack([
  ["Warhost", "The general warhost. Any craftworld unit can carry the plan."],
  ["Aspect Host", "Aspect warriors. The shrines are the army."],
  ["Guardian Battlehost", "Guardians. The militia host that wants to hold ground."],
  ["Windrider Host", "Jetbikes. A fast host that wants to disrupt."],
  ["Spirit Conclave", "Wraith constructs. The spirit host holds the line."],
  ["Seer Council", "Psykers. Farseers and warlocks as the core."],
  ["Devoted of Ynnead", "Ynnari. A host sworn to Ynnead rather than one craftworld."],
  ["Armoured Warhost", "Vehicles. A small skimmer spearhead."],
  ["Ghosts of the Webway", "Harlequins. A troupe host that wants to disrupt."],
  ["Serpent's Brood", "Harlequins pointed at a close purge."],
  ["Fateful Performance", "A small harlequin performance."],
  ["Twilight Flickers", "A small harlequin force that wants to take ground."],
  ["Path of the Outcast", "Rangers. A small outcast path."],
  ["Corsair Coterie", "Anhrathe corsairs. A coterie rather than a craftworld host."],
  ["Eldritch Raiders", "Corsairs, rangers, and shroud runners."],
]);

export const DRUKHARI_DETS = pack([
  ["Realspace Raiders", "Kabal, cult, and coven together. The full realspace raid."],
  ["Skysplinter Assault", "Transports. The raid is the disembark."],
  ["Spectacle of Spite", "Wyches. The arena fighters are the army."],
  ["Kabalite Cartel", "Archons and kabalites. A shooting cartel."],
  ["Covenite Coterie", "Haemonculus creations. Wracks, grotesques, and pain engines."],
  ["Reaper's Wager", "Drukhari and Harlequins together. Needs both sides of the wager."],
  ["Exhibition of Slaughter", "A small wych exhibition."],
  ["Kabalite Agonysts", "Blades for Hire and kabal units. A small agoniser band."],
  ["Tools of Torment", "Cronos and Talos. A small pain-engine pack."],
]);

export const VOTANN_DETS = pack([
  ["Hearthband", "Kâhls and Hearthguard. The oath-bound elite host."],
  ["Needgaârd Oathband", "A broad league host pointed at a purge."],
  ["Mercenary Oathband", "A general band that is not locked to one guild."],
  ["Brandfast Oathband", "Transports. The kin ride to the claim."],
  ["Hearthfyre Arsenal", "Memnyr and Steeljacks. The arsenal guild."],
  ["Dêlve Assault Shift", "Beserks and earthshakers. A tunnel assault."],
  ["Persecution Prospect", "Hernkyn. A prospecting band that wants to disrupt."],
  ["Armoured Trailblazers", "Sagitaurs. A small armoured trail party."],
  ["Farseekers", "Hernkyn pioneers. A small farseeking screen."],
  ["Hearthguard Covenant", "Kâhls and Einhyr. A small covenant, not the full band."],
]);

export const GSC_DETS = pack([
  ["Host of Ascension", "The whole cult. Ambush and a broad uprising."],
  ["Xenocreed Congregation", "Neophytes, acolytes, and the characters who preach to them."],
  ["Biosanctic Broodsurge", "Aberrants, genestealers, and the Biophagus."],
  ["Outlander Claw", "Vehicles and bikes. The industrial outriders."],
  ["Brood Brothers Auxilia", "Astra Militarum allies. The stolen regiment."],
  ["Final Day", "Cult and tyranid vanguard organisms together."],
  ["Heroes of the Uprising", "Kelermorph, locus, sanctus, and the reductus saboteur."],
  ["Purestrain Broodswarm", "Purestrain genestealers. A small brood."],
  ["Xenocult Masses", "Neophyte hybrids. A small mass of the outer cult."],
]);
