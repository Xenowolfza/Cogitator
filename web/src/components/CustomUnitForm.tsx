import { useState, type FormEvent } from "react";
import { melee, ranged } from "../data/helpers";
import { uid } from "../lib/id";
import type { UnitProfile, UnitRole } from "../lib/types";

const empty = {
  name: "",
  role: "other" as UnitRole,
  points: 100,
  minModels: 1,
  maxModels: 1,
  extraModelPoints: 0,
  m: '6"',
  t: "4",
  sv: "3+",
  w: "2",
  ld: "6+",
  oc: "1",
  invuln: "",
  keywords: "INFANTRY",
  factionKeywords: "",
  composition: "1 model",
  abilityName: "",
  abilityText: "",
  rangedName: "",
  rangedRange: '24"',
  rangedA: "1",
  rangedBs: "3+",
  rangedS: "4",
  rangedAp: "0",
  rangedD: "1",
  meleeName: "Close combat weapon",
  meleeA: "3",
  meleeWs: "3+",
  meleeS: "4",
  meleeAp: "0",
  meleeD: "1",
};

type Props = {
  factionId: string;
  factionName: string;
  onCancel: () => void;
  onSave: (profile: UnitProfile) => void;
};

export function CustomUnitForm({ factionId, factionName, onCancel, onSave }: Props) {
  const [form, setForm] = useState(empty);
  const set = (key: keyof typeof empty, value: string | number) =>
    setForm((f) => ({ ...f, [key]: value }));

  function submit(e: FormEvent) {
    e.preventDefault();
    if (!form.name.trim()) return;

    const weapons = [];
    if (form.rangedName.trim()) {
      weapons.push(
        ranged(
          form.rangedName.trim(),
          form.rangedRange,
          form.rangedA,
          form.rangedBs,
          form.rangedS,
          form.rangedAp,
          form.rangedD,
        ),
      );
    }
    if (form.meleeName.trim()) {
      weapons.push(
        melee(form.meleeName.trim(), form.meleeA, form.meleeWs, form.meleeS, form.meleeAp, form.meleeD),
      );
    }

    const min = Math.max(1, Number(form.minModels) || 1);
    const max = Math.max(min, Number(form.maxModels) || min);

    onSave({
      id: `custom-${uid()}`,
      name: form.name.trim(),
      factionId,
      role: form.role,
      points: Number(form.points) || 0,
      pointsMode: "unit",
      extraModelPoints: Number(form.extraModelPoints) || 0,
      minModels: min,
      maxModels: max,
      stats: {
        m: form.m,
        t: form.t,
        sv: form.sv,
        w: form.w,
        ld: form.ld,
        oc: form.oc,
        invuln: form.invuln.trim() || undefined,
      },
      keywords: form.keywords
        .split(",")
        .map((k) => k.trim())
        .filter(Boolean),
      factionKeywords: (form.factionKeywords || factionName)
        .split(",")
        .map((k) => k.trim())
        .filter(Boolean),
      coreAbilities: [],
      weapons,
      wargearOptions: [],
      abilities: form.abilityName.trim()
        ? [{ name: form.abilityName.trim(), text: form.abilityText.trim() }]
        : [],
      composition: form.composition.trim() || `${min} model${min === 1 ? "" : "s"}`,
    });
  }

  return (
    <form className="custom-form" onSubmit={submit}>
      <h3>Custom unit</h3>
      <p className="muted">Enter stats from your own book. Saved only on this list.</p>

      <label>
        Name
        <input value={form.name} onChange={(e) => set("name", e.target.value)} required />
      </label>

      <div className="form-row">
        <label>
          Role
          <select value={form.role} onChange={(e) => set("role", e.target.value as UnitRole)}>
            <option value="character">Character</option>
            <option value="battleline">Battleline</option>
            <option value="dedicatedTransport">Dedicated Transport</option>
            <option value="other">Other</option>
          </select>
        </label>
        <label>
          Points (unit)
          <input type="number" min={0} value={form.points} onChange={(e) => set("points", Number(e.target.value))} />
        </label>
      </div>

      <div className="form-row">
        <label>
          Min models
          <input type="number" min={1} value={form.minModels} onChange={(e) => set("minModels", Number(e.target.value))} />
        </label>
        <label>
          Max models
          <input type="number" min={1} value={form.maxModels} onChange={(e) => set("maxModels", Number(e.target.value))} />
        </label>
        <label>
          Extra model pts
          <input
            type="number"
            min={0}
            value={form.extraModelPoints}
            onChange={(e) => set("extraModelPoints", Number(e.target.value))}
          />
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
            <input value={form[key]} onChange={(e) => set(key, e.target.value)} />
          </label>
        ))}
      </div>

      <label>
        Keywords (comma-separated)
        <input value={form.keywords} onChange={(e) => set("keywords", e.target.value)} />
      </label>

      <fieldset>
        <legend>Ranged weapon (optional)</legend>
        <label>
          Name
          <input value={form.rangedName} onChange={(e) => set("rangedName", e.target.value)} />
        </label>
        <div className="form-row compact">
          <label>
            Range
            <input value={form.rangedRange} onChange={(e) => set("rangedRange", e.target.value)} />
          </label>
          <label>
            A
            <input value={form.rangedA} onChange={(e) => set("rangedA", e.target.value)} />
          </label>
          <label>
            BS
            <input value={form.rangedBs} onChange={(e) => set("rangedBs", e.target.value)} />
          </label>
          <label>
            S
            <input value={form.rangedS} onChange={(e) => set("rangedS", e.target.value)} />
          </label>
          <label>
            AP
            <input value={form.rangedAp} onChange={(e) => set("rangedAp", e.target.value)} />
          </label>
          <label>
            D
            <input value={form.rangedD} onChange={(e) => set("rangedD", e.target.value)} />
          </label>
        </div>
      </fieldset>

      <fieldset>
        <legend>Melee weapon</legend>
        <label>
          Name
          <input value={form.meleeName} onChange={(e) => set("meleeName", e.target.value)} />
        </label>
        <div className="form-row compact">
          <label>
            A
            <input value={form.meleeA} onChange={(e) => set("meleeA", e.target.value)} />
          </label>
          <label>
            WS
            <input value={form.meleeWs} onChange={(e) => set("meleeWs", e.target.value)} />
          </label>
          <label>
            S
            <input value={form.meleeS} onChange={(e) => set("meleeS", e.target.value)} />
          </label>
          <label>
            AP
            <input value={form.meleeAp} onChange={(e) => set("meleeAp", e.target.value)} />
          </label>
          <label>
            D
            <input value={form.meleeD} onChange={(e) => set("meleeD", e.target.value)} />
          </label>
        </div>
      </fieldset>

      <label>
        Ability name
        <input value={form.abilityName} onChange={(e) => set("abilityName", e.target.value)} />
      </label>
      <label>
        Ability text
        <textarea value={form.abilityText} onChange={(e) => set("abilityText", e.target.value)} rows={3} />
      </label>
      <label>
        Composition
        <input value={form.composition} onChange={(e) => set("composition", e.target.value)} />
      </label>

      <div className="form-actions">
        <button type="button" className="btn ghost" onClick={onCancel}>
          Cancel
        </button>
        <button type="submit" className="btn gold">
          Add to list
        </button>
      </div>
    </form>
  );
}
