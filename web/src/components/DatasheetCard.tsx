import type { CardThemeId, ResolvedUnit, Weapon } from "../lib/types";

function WeaponTable({ title, weapons, skillLabel }: { title: string; weapons: Weapon[]; skillLabel: string }) {
  if (weapons.length === 0) return null;
  return (
    <table className="wep-table">
      <thead>
        <tr>
          <th>{title}</th>
          <th>RANGE</th>
          <th>A</th>
          <th>{skillLabel}</th>
          <th>S</th>
          <th>AP</th>
          <th>D</th>
        </tr>
      </thead>
      <tbody>
        {weapons.map((w, i) => (
          <tr key={`${w.name}-${i}`}>
            <td>
              {w.name}
              {w.keywords.length > 0 && (
                <div className="wep-keys">[{w.keywords.join("] [")}]</div>
              )}
            </td>
            <td>{w.range}</td>
            <td>{w.attacks}</td>
            <td>{w.skill}</td>
            <td>{w.strength}</td>
            <td>{w.ap}</td>
            <td>{w.damage}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function DatasheetCard({ unit, theme = "parchment" }: { unit: ResolvedUnit; theme?: CardThemeId }) {
  const { profile, selection, weapons, points, enhancement } = unit;
  const ranged = weapons.filter((w) => w.kind === "ranged");
  const melee = weapons.filter((w) => w.kind === "melee");
  const models =
    selection.models !== profile.minModels || profile.minModels !== profile.maxModels
      ? `${selection.models} models`
      : null;

  return (
    <article className="sheet" data-theme={theme}>
      <header className="sheet-head">
        <div>
          <div className="sheet-kicker">{profile.factionKeywords.join(" · ")}</div>
          <h2 className="sheet-name">{profile.name}</h2>
        </div>
        {profile.stats.invuln && (
          <div className="sheet-inv">
            <span>INV</span>
            <strong>{profile.stats.invuln}</strong>
          </div>
        )}
      </header>

      <div className="statline">
        {(
          [
            ["M", profile.stats.m],
            ["T", profile.stats.t],
            ["SV", profile.stats.sv],
            ["W", profile.stats.w],
            ["LD", profile.stats.ld],
            ["OC", profile.stats.oc],
          ] as const
        ).map(([label, value]) => (
          <div className="stat" key={label}>
            <span>{label}</span>
            <b>{value}</b>
          </div>
        ))}
      </div>

      <WeaponTable title="RANGED WEAPONS" weapons={ranged} skillLabel="BS" />
      <WeaponTable title="MELEE WEAPONS" weapons={melee} skillLabel="WS" />

      <section className="sheet-abil">
        {profile.coreAbilities.length > 0 && (
          <p>
            <span className="tag">CORE</span> {profile.coreAbilities.join(", ")}
          </p>
        )}
        {profile.factionAbility && (
          <p>
            <span className="tag">FACTION</span> {profile.factionAbility}
          </p>
        )}
        {profile.abilities.map((a) => (
          <p key={a.name}>
            <strong>{a.name}:</strong> {a.text}
          </p>
        ))}
        {enhancement && (
          <p>
            <span className="tag">ENHANCEMENT</span>{" "}
            <strong>{enhancement.name}</strong> ({enhancement.points} pts) — {enhancement.text}
          </p>
        )}
        {profile.damaged && (
          <p className="damaged">
            <strong>DAMAGED: {profile.damaged.threshold}.</strong> {profile.damaged.profile}
          </p>
        )}
      </section>

      <footer className="sheet-foot">
        <div>
          <strong>UNIT COMPOSITION</strong>
          <div>
            {profile.composition}
            {models ? ` · Fielded: ${models}` : ""}
          </div>
        </div>
        <div className="sheet-pts">{points} PTS</div>
      </footer>
      <div className="sheet-keys">
        <strong>KEYWORDS:</strong> {profile.keywords.join(", ")}
      </div>
    </article>
  );
}
