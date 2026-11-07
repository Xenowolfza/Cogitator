import { DatasheetCard } from "./DatasheetCard";
import { detachmentById, factionById } from "../data/factions";
import { listPoints } from "../lib/points";
import type { ArmyList, ResolvedUnit } from "../lib/types";

type Props = {
  list: ArmyList;
  units: ResolvedUnit[];
};

export function DatasheetsView({ list, units }: Props) {
  const faction = factionById(list.factionId);
  const detachment = detachmentById(list.factionId, list.detachmentId);
  const pts = listPoints(list);

  return (
    <div className="datasheets">
      <aside className="roster print-block">
        <div className="roster-head">
          <p className="kicker">Army roster</p>
          <h2>{list.name}</h2>
          <p>
            {faction?.name} · {detachment?.name}
          </p>
          <p className="roster-pts">
            {pts} / {list.pointsCap} pts
          </p>
        </div>
        {detachment && <p className="roster-det">{detachment.summary}</p>}
        <ol className="roster-list">
          {units.map((u) => (
            <li key={u.selection.uid}>
              <span>
                {u.profile.name}
                {u.profile.minModels !== u.profile.maxModels ? ` ×${u.selection.models}` : ""}
                {u.enhancement ? ` + ${u.enhancement.name}` : ""}
              </span>
              <span>{u.points}</span>
            </li>
          ))}
        </ol>
        {units.length === 0 && <p className="muted">No units in this list yet.</p>}
      </aside>

      <div className="sheet-grid">
        {units.map((unit) => (
          <DatasheetCard key={unit.selection.uid} unit={unit} theme={list.cardTheme ?? "parchment"} />
        ))}
      </div>
    </div>
  );
}
