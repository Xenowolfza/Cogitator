import { useState, type FormEvent } from "react";
import { melee, ranged } from "../data/helpers";
import { uid } from "../lib/id";
import type { Ability, UnitProfile, UnitRole, Weapon } from "../lib/types";

type Props = {
  factionId: string;
  factionName: string;
  existing?: UnitProfile;
  onCancel: () => void;
  onSave: (profile: UnitProfile) => void;
};

function blankWeapon(kind: Weapon["kind"]): Weapon {
  return kind === "ranged"
    ? ranged("New ranged weapon", '24"', "1", "3+", "4", "0", "1")
    : melee("Close combat weapon", "3", "3+", "4", "0", "1");
}

export function UnitEditor({ factionId, factionName, existing, onCancel, onSave }: Props) {
  const [name, setName] = useState(existing?.name ?? "");
  const [role, setRole] = useState<UnitRole>(existing?.role ?? "other");
  const [points, setPoints] = useState(existing?.points ?? 100);
  const [minModels, setMinModels] = useState(existing?.minModels ?? 1);
  const [maxModels, setMaxModels] = useState(existing?.maxModels ?? 1);
  const [extraModelPoints, setExtraModelPoints] = useState(existing?.extraModelPoints ?? 0);
  const [stats, setStats] = useState({
    m: existing?.stats.m ?? '6"',
    t: existing?.stats.t ?? "4",
    sv: existing?.stats.sv ?? "3+",
    w: existing?.stats.w ?? "2",
    ld: existing?.stats.ld ?? "6+",
    oc: existing?.stats.oc ?? "1",
    invuln: existing?.stats.invuln ?? "",
  });
  const [keywords, setKeywords] = useState(existing?.keywords.join(", ") ?? "INFANTRY");
  const [factionKeywords, setFactionKeywords] = useState(
    existing?.factionKeywords.join(", ") ?? factionName,
  );
  const [coreAbilities, setCoreAbilities] = useState(existing?.coreAbilities.join(", ") ?? "");
  const [factionAbility, setFactionAbility] = useState(existing?.factionAbility ?? "");
  const [composition, setComposition] = useState(existing?.composition ?? "1 model");
  const [weapons, setWeapons] = useState<Weapon[]>(
    existing?.weapons.length ? existing.weapons.map((w) => ({ ...w, keywords: [...w.keywords] })) : [blankWeapon("melee")],
  );
  const [abilities, setAbilities] = useState<Ability[]>(
    existing?.abilities.length ? existing.abilities.map((a) => ({ ...a })) : [{ name: "", text: "" }],
  );

  function setWeapon(index: number, patch: Partial<Weapon>) {
    setWeapons((list) => list.map((w, i) => (i === index ? { ...w, ...patch } : w)));
  }

  function submit(e: FormEvent) {
    e.preventDefault();
    if (!name.trim()) return;
    const min = Math.max(1, Number(minModels) || 1);
    const max = Math.max(min, Number(maxModels) || min);
    onSave({
      id: existing?.id ?? `custom-${uid()}`,
      name: name.trim(),
      factionId: existing?.factionId ?? factionId,
      role,
      points: Number(points) || 0,
      pointsMode: existing?.pointsMode ?? "unit",
      extraModelPoints: Number(extraModelPoints) || 0,
      minModels: min,
      maxModels: max,
      stats: {
        m: stats.m,
        t: stats.t,
        sv: stats.sv,
        w: stats.w,
        ld: stats.ld,
        oc: stats.oc,
        invuln: stats.invuln.trim() || undefined,
      },
      keywords: keywords.split(",").map((k) => k.trim()).filter(Boolean),
      factionKeywords: (factionKeywords || factionName).split(",").map((k) => k.trim()).filter(Boolean),
      coreAbilities: coreAbilities.split(",").map((k) => k.trim()).filter(Boolean),
      factionAbility: factionAbility.trim() || undefined,
      weapons: weapons.filter((w) => w.name.trim()),
      wargearOptions: existing?.wargearOptions ?? [],
      abilities: abilities.filter((a) => a.name.trim()),
      composition: composition.trim() || `${min} model${min === 1 ? "" : "s"}`,
      damaged: existing?.damaged,
    });
  }

  return (
    <form className="custom-form" onSubmit={submit}>
      <h3>{existing ? "Edit unit" : "Custom unit"}</h3>
      <p className="muted">
        {existing
          ? "Changes apply only to this entry on the list. Wargear options are kept."
          : "Enter stats from your own book. Saved only on this list."}
      </p>

      <label>
        Name
        <input value={name} onChange={(e) => setName(e.target.value)} required />
      </label>

      <div className="form-row">
        <label>
          Role
          <select value={role} onChange={(e) => setRole(e.target.value as UnitRole)}>
            <option value="character">Character</option>
            <option value="battleline">Battleline</option>
            <option value="dedicatedTransport">Dedicated Transport</option>
            <option value="other">Other</option>
          </select>
        </label>
        <label>
          Points
          <input type="number" min={0} value={points} onChange={(e) => setPoints(Number(e.target.value))} />
        </label>
      </div>

      <div className="form-row">
        <label>
          Min models
          <input type="number" min={1} value={minModels} onChange={(e) => setMinModels(Number(e.target.value))} />
        </label>
        <label>
          Max models
          <input type="number" min={1} value={maxModels} onChange={(e) => setMaxModels(Number(e.target.value))} />
        </label>
        <label>
          Extra model pts
          <input type="number" min={0} value={extraModelPoints} onChange={(e) => setExtraModelPoints(Number(e.target.value))} />
        </label>
      </div>

      <div className="stat-inputs">
        {(
          [
            ["m", "M"],
            ["t", "T"],
            ["sv", "SV"],
            ["w", "W"],
            ["ld", "LD"],
            ["oc", "OC"],
            ["invuln", "INV"],
          ] as const
        ).map(([key, label]) => (
          <label key={key}>
            {label}
            <input value={stats[key]} onChange={(e) => setStats((s) => ({ ...s, [key]: e.target.value }))} />
          </label>
        ))}
      </div>

      <label>
        Keywords (comma-separated)
        <input value={keywords} onChange={(e) => setKeywords(e.target.value)} />
      </label>
      <label>
        Faction keywords
        <input value={factionKeywords} onChange={(e) => setFactionKeywords(e.target.value)} />
      </label>
      <label>
        Core abilities
        <input value={coreAbilities} onChange={(e) => setCoreAbilities(e.target.value)} />
      </label>
      <label>
        Faction ability
        <input value={factionAbility} onChange={(e) => setFactionAbility(e.target.value)} />
      </label>
      <label>
        Composition
        <input value={composition} onChange={(e) => setComposition(e.target.value)} />
      </label>

      <fieldset>
        <legend>Weapons</legend>
        <div className="weapon-editor">
          {weapons.map((w, i) => (
            <div className="weapon-row" key={i}>
              <div className="form-row">
                <label>
                  Name
                  <input value={w.name} onChange={(e) => setWeapon(i, { name: e.target.value })} />
                </label>
                <label>
                  Kind
                  <select
                    value={w.kind}
                    onChange={(e) => {
                      const kind = e.target.value as Weapon["kind"];
                      setWeapon(i, { kind, range: kind === "melee" ? "Melee" : w.range === "Melee" ? '24"' : w.range });
                    }}
                  >
                    <option value="ranged">Ranged</option>
                    <option value="melee">Melee</option>
                  </select>
                </label>
              </div>
              <div className="form-row compact">
                <label>
                  Range
                  <input value={w.range} onChange={(e) => setWeapon(i, { range: e.target.value })} />
                </label>
                <label>
                  A
                  <input value={w.attacks} onChange={(e) => setWeapon(i, { attacks: e.target.value })} />
                </label>
                <label>
                  BS/WS
                  <input value={w.skill} onChange={(e) => setWeapon(i, { skill: e.target.value })} />
                </label>
                <label>
                  S
                  <input value={w.strength} onChange={(e) => setWeapon(i, { strength: e.target.value })} />
                </label>
                <label>
                  AP
                  <input value={w.ap} onChange={(e) => setWeapon(i, { ap: e.target.value })} />
                </label>
                <label>
                  D
                  <input value={w.damage} onChange={(e) => setWeapon(i, { damage: e.target.value })} />
                </label>
              </div>
              <label>
                Weapon keywords (comma-separated)
                <input
                  value={w.keywords.join(", ")}
                  onChange={(e) =>
                    setWeapon(i, {
                      keywords: e.target.value.split(",").map((k) => k.trim()).filter(Boolean),
                    })
                  }
                />
              </label>
              <button type="button" className="btn ghost slim" onClick={() => setWeapons((list) => list.filter((_, idx) => idx !== i))}>
                Remove weapon
              </button>
            </div>
          ))}
        </div>
        <div className="form-actions">
          <button type="button" className="btn ghost slim" onClick={() => setWeapons((list) => [...list, blankWeapon("ranged")])}>
            Add ranged
          </button>
          <button type="button" className="btn ghost slim" onClick={() => setWeapons((list) => [...list, blankWeapon("melee")])}>
            Add melee
          </button>
        </div>
      </fieldset>

      <fieldset>
        <legend>Abilities</legend>
        {abilities.map((a, i) => (
          <div className="weapon-row" key={i}>
            <label>
              Name
              <input
                value={a.name}
                onChange={(e) =>
                  setAbilities((list) => list.map((item, idx) => (idx === i ? { ...item, name: e.target.value } : item)))
                }
              />
            </label>
            <label>
              Text
              <textarea
                rows={2}
                value={a.text}
                onChange={(e) =>
                  setAbilities((list) => list.map((item, idx) => (idx === i ? { ...item, text: e.target.value } : item)))
                }
              />
            </label>
            <button type="button" className="btn ghost slim" onClick={() => setAbilities((list) => list.filter((_, idx) => idx !== i))}>
              Remove ability
            </button>
          </div>
        ))}
        <button type="button" className="btn ghost slim" onClick={() => setAbilities((list) => [...list, { name: "", text: "" }])}>
          Add ability
        </button>
      </fieldset>

      <div className="form-actions">
        <button type="button" className="btn ghost" onClick={onCancel}>
          Cancel
        </button>
        <button type="submit" className="btn gold">
          {existing ? "Save changes" : "Add to list"}
        </button>
      </div>
    </form>
  );
}
