import { useMemo, useState } from "react";
import { unitsForFaction } from "../data/catalog";
import { factionById } from "../data/factions";
import type { UnitProfile, UnitRole } from "../lib/types";

const ROLES: { id: "all" | UnitRole; label: string }[] = [
  { id: "all", label: "All" },
  { id: "character", label: "Character" },
  { id: "battleline", label: "Battleline" },
  { id: "dedicatedTransport", label: "Transport" },
  { id: "other", label: "Other" },
];

function roleLabel(role: UnitRole): string {
  if (role === "dedicatedTransport") return "Transport";
  return role[0].toUpperCase() + role.slice(1);
}

function pointsLabel(unit: UnitProfile): string {
  if (unit.pointsMode === "model") return `${unit.points} pts/model`;
  if (unit.minModels !== unit.maxModels && unit.extraModelPoints) {
    return `${unit.points} pts (${unit.minModels}–${unit.maxModels})`;
  }
  return `${unit.points} pts`;
}

type Props = {
  factionId: string;
  onAdd: (unit: UnitProfile) => void;
  onCustom: () => void;
};

export function CatalogPanel({ factionId, onAdd, onCustom }: Props) {
  const [query, setQuery] = useState("");
  const [role, setRole] = useState<(typeof ROLES)[number]["id"]>("all");
  const faction = factionById(factionId);
  const includedNames = (faction?.includeFactionIds ?? [])
    .map((id) => factionById(id)?.name)
    .filter(Boolean);

  const units = useMemo(() => {
    const q = query.trim().toLowerCase();
    return unitsForFaction(factionId).filter((u) => {
      if (role !== "all" && u.role !== role) return false;
      if (!q) return true;
      const hay = `${u.name} ${u.keywords.join(" ")} ${u.factionKeywords.join(" ")}`.toLowerCase();
      return hay.includes(q);
    });
  }, [factionId, query, role]);

  return (
    <section className="catalog">
      <div className="panel-head">
        <h2>Catalogue</h2>
        <button type="button" className="btn ghost" onClick={onCustom}>
          Custom unit
        </button>
      </div>

      {includedNames.length > 0 && (
        <p className="muted catalog-note">
          {faction?.name} models plus shared {includedNames.join(", ")} units.
        </p>
      )}

      <input
        className="search"
        placeholder="Search units or keywords"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />

      <div className="chips">
        {ROLES.map((r) => (
          <button
            key={r.id}
            type="button"
            className={role === r.id ? "chip on" : "chip"}
            onClick={() => setRole(r.id)}
          >
            {r.label}
          </button>
        ))}
      </div>

      <ul className="unit-grid">
        {units.map((unit) => (
          <li key={unit.id} className="unit-card">
            <div className="unit-card-top">
              <div>
                <div className="unit-role">
                  {roleLabel(unit.role)}
                  {unit.factionId !== factionId ? " · shared" : ""}
                </div>
                <h3>{unit.name}</h3>
              </div>
              <div className="unit-pts">{pointsLabel(unit)}</div>
            </div>
            <div className="mini-stats">
              <span>M {unit.stats.m}</span>
              <span>T {unit.stats.t}</span>
              <span>SV {unit.stats.sv}</span>
              <span>W {unit.stats.w}</span>
              <span>LD {unit.stats.ld}</span>
              <span>OC {unit.stats.oc}</span>
            </div>
            <button type="button" className="btn gold slim" onClick={() => onAdd(unit)}>
              Add
            </button>
          </li>
        ))}
      </ul>
      {units.length === 0 && <p className="muted">No units match that filter.</p>}
    </section>
  );
}
