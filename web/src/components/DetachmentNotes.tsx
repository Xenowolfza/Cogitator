import { useState, type FormEvent } from "react";
import { uid } from "../lib/id";
import { deleteReference, openReference, saveReference } from "../lib/references";
import type { Detachment, Enhancement } from "../lib/types";

const MAX_BYTES = 25 * 1024 * 1024;

type Props = {
  factionName: string;
  notes: Detachment[];
  onChange: (notes: Detachment[]) => void;
};

function blankEnhancement(): Enhancement {
  return { id: uid(), name: "", points: 0, text: "" };
}

export function DetachmentNotes({ factionName, notes, onChange }: Props) {
  const [open, setOpen] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [summary, setSummary] = useState("");
  const [enhancements, setEnhancements] = useState<Enhancement[]>([blankEnhancement()]);
  const [file, setFile] = useState<File | null>(null);
  const [keptRef, setKeptRef] = useState<{ id: string; name: string } | null>(null);
  const [error, setError] = useState("");

  function reset() {
    setEditingId(null);
    setName("");
    setSummary("");
    setEnhancements([blankEnhancement()]);
    setFile(null);
    setKeptRef(null);
    setError("");
  }

  function edit(note: Detachment) {
    setOpen(true);
    setEditingId(note.id);
    setName(note.name);
    setSummary(note.summary);
    setEnhancements(note.enhancements.length ? note.enhancements.map((e) => ({ ...e })) : [blankEnhancement()]);
    setFile(null);
    setKeptRef(note.referenceId && note.referenceName ? { id: note.referenceId, name: note.referenceName } : null);
    setError("");
  }

  async function remove(note: Detachment) {
    if (note.referenceId) await deleteReference(note.referenceId).catch(() => undefined);
    onChange(notes.filter((n) => n.id !== note.id));
    if (editingId === note.id) reset();
  }

  async function submit(e: FormEvent) {
    e.preventDefault();
    setError("");
    if (!name.trim() || !summary.trim()) {
      setError("Add a name and your own summary.");
      return;
    }
    if (file && file.size > MAX_BYTES) {
      setError("That file is over 25 MB. Keep the pack on disk and type a short note instead.");
      return;
    }
    const id = editingId ?? `custom-${uid()}`;
    let referenceId = keptRef?.id;
    let referenceName = keptRef?.name;
    if (file) {
      referenceId = id;
      referenceName = file.name;
      await saveReference(id, file);
    }
    const next: Detachment = {
      id,
      name: name.trim(),
      summary: summary.trim(),
      custom: true,
      referenceId,
      referenceName,
      enhancements: enhancements
        .filter((enh) => enh.name.trim())
        .map((enh) => ({ ...enh, name: enh.name.trim(), text: enh.text.trim(), points: Number(enh.points) || 0 })),
    };
    onChange(editingId ? notes.map((n) => (n.id === editingId ? next : n)) : [...notes, next]);
    reset();
  }

  return (
    <details className="det-tools no-print" open={open} onToggle={(e) => setOpen((e.target as HTMLDetailsElement).open)}>
      <summary>Your detachment notes</summary>
      <p className="muted">
        Built-in detachments are short playstyle notes for {factionName}, using public 11th-edition names from mid-2026.
        Official wording is not copied here. If your faction pack is newer, write your own note. You can attach a PDF as a
        private file in this browser — the app does not read or extract it. Packs are listed on{" "}
        <a className="det-link" href="https://www.warhammer-community.com/en-gb/downloads/warhammer-40000/" target="_blank" rel="noreferrer">
          Warhammer Community downloads
        </a>
        .
      </p>

      {notes.length > 0 && (
        <ul className="custom-det-list">
          {notes.map((note) => (
            <li key={note.id}>
              <span>
                {note.name}
                {note.referenceName ? ` · ${note.referenceName}` : ""}
              </span>
              <span className="sel-actions">
                {note.referenceId && (
                  <button type="button" className="btn ghost slim" onClick={() => void openReference(note.referenceId!)}>
                    Open file
                  </button>
                )}
                <button type="button" className="btn ghost slim" onClick={() => edit(note)}>
                  Edit
                </button>
                <button type="button" className="btn danger slim" onClick={() => void remove(note)}>
                  Delete
                </button>
              </span>
            </li>
          ))}
        </ul>
      )}

      <form className="custom-form" onSubmit={(e) => void submit(e)}>
        <label>
          Name
          <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Detachment name" />
        </label>
        <label>
          Your summary
          <textarea rows={3} value={summary} onChange={(e) => setSummary(e.target.value)} placeholder="Write the rule in your own words" />
        </label>
        {enhancements.map((enh, i) => (
          <div className="weapon-row" key={enh.id}>
            <div className="form-row">
              <label>
                Enhancement
                <input
                  value={enh.name}
                  onChange={(e) =>
                    setEnhancements((list) => list.map((item, idx) => (idx === i ? { ...item, name: e.target.value } : item)))
                  }
                />
              </label>
              <label>
                Points
                <input
                  type="number"
                  min={0}
                  value={enh.points}
                  onChange={(e) =>
                    setEnhancements((list) =>
                      list.map((item, idx) => (idx === i ? { ...item, points: Number(e.target.value) } : item)),
                    )
                  }
                />
              </label>
            </div>
            <label>
              Text you typed
              <textarea
                rows={2}
                value={enh.text}
                onChange={(e) =>
                  setEnhancements((list) => list.map((item, idx) => (idx === i ? { ...item, text: e.target.value } : item)))
                }
              />
            </label>
          </div>
        ))}
        <div className="form-actions">
          <button type="button" className="btn ghost slim" onClick={() => setEnhancements((list) => [...list, blankEnhancement()])}>
            Add enhancement
          </button>
        </div>
        <label>
          Attach faction pack (optional, stays on this device)
          <input
            type="file"
            accept="application/pdf,.pdf"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
        </label>
        {keptRef && !file && <p className="muted">Keeping attached file: {keptRef.name}</p>}
        {error && <p className="warn">{error}</p>}
        <div className="form-actions">
          {editingId && (
            <button type="button" className="btn ghost" onClick={reset}>
              Cancel
            </button>
          )}
          <button type="submit" className="btn gold">
            {editingId ? "Save note" : "Add note"}
          </button>
        </div>
      </form>
    </details>
  );
}
