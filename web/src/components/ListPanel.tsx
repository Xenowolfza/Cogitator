import { detachmentById, extrasFor } from "../data/factions";
import { selectionPoints } from "../lib/points";
import { toggleOption } from "../lib/resolve";
import type { ArmyList, UnitProfile, UnitSelection } from "../lib/types";

function roleLabel(role: UnitProfile["role"]): string {
  if (role === "dedicatedTransport") return "Transport";
  return role[0].toUpperCase() + role.slice(1);
}

type Props = {
  list: ArmyList;
  profileFor: (sel: UnitSelection) => UnitProfile | undefined;
  onChange: (selections: UnitSelection[]) => void;
  onEdit: (sel: UnitSelection) => void;
};

export function ListPanel({ list, profileFor, onChange, onEdit }: Props) {
  const detachment = detachmentById(list.factionId, list.detachmentId, extrasFor(list));
  const usedEnhancements = new Set(
    list.selections.map((s) => s.enhancementId).filter((id): id is string => Boolean(id)),
  );

  function update(uid: string, patch: Partial<UnitSelection>) {
    onChange(list.selections.map((s) => (s.uid === uid ? { ...s, ...patch } : s)));
  }

  function remove(uid: string) {
    onChange(list.selections.filter((s) => s.uid !== uid));
  }

  function duplicate(sel: UnitSelection) {
    onChange([
      ...list.selections,
      { ...sel, uid: crypto.randomUUID(), enhancementId: undefined },
    ]);
  }

  return (
    <section className="army">
      <div className="panel-head">
        <h2>Army list</h2>
        <span className="count">{list.selections.length} units</span>
      </div>

      {list.selections.length === 0 && (
        <p className="muted">Add units from the catalogue to start a list.</p>
      )}

      <ul className="sel-list">
        {list.selections.map((sel) => {
          const profile = profileFor(sel);
          if (!profile) return null;
          const pts = selectionPoints(sel, list);
          const groups = new Map<string, string[]>();
          for (const opt of profile.wargearOptions) {
            if (!opt.group) continue;
            const ids = groups.get(opt.group) ?? [];
            ids.push(opt.id);
            groups.set(opt.group, ids);
          }

          return (
            <li key={sel.uid} className="sel-card">
              <div className="sel-top">
                <div>
                  <div className="unit-role">{roleLabel(profile.role)}</div>
                  <h3>
                    {profile.name}
                    {sel.customProfile ? <span className="edited-tag">Edited</span> : null}
                  </h3>
                </div>
                <div className="unit-pts">{pts} pts</div>
              </div>

              {profile.maxModels > profile.minModels && (
                <div className="stepper">
                  <span>Models</span>
                  <button
                    type="button"
                    onClick={() => update(sel.uid, { models: Math.max(profile.minModels, sel.models - 1) })}
                    disabled={sel.models <= profile.minModels}
                  >
                    −
                  </button>
                  <strong>
                    {sel.models}/{profile.maxModels}
                  </strong>
                  <button
                    type="button"
                    onClick={() => update(sel.uid, { models: Math.min(profile.maxModels, sel.models + 1) })}
                    disabled={sel.models >= profile.maxModels}
                  >
                    +
                  </button>
                </div>
              )}

              {profile.wargearOptions.length > 0 && (
                <div className="options">
                  {profile.wargearOptions.map((opt) => {
                    const checked = sel.optionIds.includes(opt.id);
                    const groupIds = opt.group ? groups.get(opt.group) : undefined;
                    return (
                      <label key={opt.id} className="opt">
                        <input
                          type="checkbox"
                          name={`${sel.uid}-${opt.id}`}
                          checked={checked}
                          onChange={() =>
                            update(sel.uid, {
                              optionIds: toggleOption(sel.optionIds, opt.id, opt.group, groupIds),
                            })
                          }
                        />
                        <span>
                          {opt.label}
                          {opt.points ? ` (+${opt.points})` : ""}
                        </span>
                      </label>
                    );
                  })}
                </div>
              )}

              {profile.role === "character" && detachment && detachment.enhancements.length > 0 && (
                <label className="enh">
                  Enhancement
                  <select
                    value={sel.enhancementId ?? ""}
                    onChange={(e) =>
                      update(sel.uid, { enhancementId: e.target.value || undefined })
                    }
                  >
                    <option value="">None</option>
                    {detachment.enhancements.map((enh) => (
                      <option
                        key={enh.id}
                        value={enh.id}
                        disabled={usedEnhancements.has(enh.id) && sel.enhancementId !== enh.id}
                      >
                        {enh.name} ({enh.points} pts)
                      </option>
                    ))}
                  </select>
                </label>
              )}

              <div className="sel-actions">
                <button type="button" className="btn ghost slim" onClick={() => onEdit(sel)}>
                  Edit stats
                </button>
                {sel.customProfile && !sel.unitId.startsWith("custom-") && (
                  <button
                    type="button"
                    className="btn ghost slim"
                    onClick={() => update(sel.uid, { customProfile: undefined })}
                  >
                    Revert stats
                  </button>
                )}
                <button type="button" className="btn ghost slim" onClick={() => duplicate(sel)}>
                  Duplicate
                </button>
                <button type="button" className="btn danger slim" onClick={() => remove(sel.uid)}>
                  Remove
                </button>
              </div>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
