import { ArrowRight, Camera, Check, FolderOpen, Pencil, Plus, Search, X } from 'lucide-react';
import { useState } from 'react';
import type { Checkpoint } from './types';

type Props = {
  checkpoints: Checkpoint[];
  busy: boolean;
  onOpen: (id: string, target: 'admin' | 'run') => void;
  onUpdate: (checkpoint: Checkpoint, changes: Partial<Checkpoint>) => void;
};

export default function CheckpointLibrary({ checkpoints, busy, onOpen, onUpdate }: Props) {
  const [query, setQuery] = useState('');
  const [visibility, setVisibility] = useState('all');
  const search = query.trim().toLocaleLowerCase();
  const filtered = checkpoints.filter(checkpoint =>
    [checkpoint.name, checkpoint.task, checkpoint.notes].join(' ').toLocaleLowerCase().includes(search)
    && (visibility === 'all' || checkpoint.visible === (visibility === 'visible'))
  );

  return <main className="checkpoint-library">
    <div className="library-intro">
      <div>
        <h1>Library <span className="library-count" aria-hidden="true">{checkpoints.length}</span></h1>
        <p>Your saved plans. Pick one up, adapt it to a new scene, and run it.</p>
      </div>
      <a className="button-link" href="/admin"><Plus size={16} aria-hidden="true" />Prepare a plan</a>
    </div>

    {checkpoints.length > 0 && <div className="library-filters">
      <div className="library-search">
        <Search size={17} aria-hidden="true" />
        <input type="search" aria-label="Search saved plans" placeholder="Search plans, tasks, or notes…"
          value={query} onChange={event => setQuery(event.target.value)} />
        {query && <button className="library-clear" aria-label="Clear search" onClick={() => setQuery('')}>
          <X size={15} aria-hidden="true" />
        </button>}
      </div>
      <select aria-label="Filter plans by visibility" value={visibility} onChange={event => setVisibility(event.target.value)}>
        <option value="all">All plans</option>
        <option value="visible">Available in Run</option>
        <option value="hidden">Hidden from Run</option>
      </select>
      <span className="library-results" role="status">{filtered.length} {filtered.length === 1 ? 'plan' : 'plans'}{search || visibility !== 'all' ? ' found' : ' saved'}</span>
    </div>}

    <div className="checkpoint-list">{filtered.map(checkpoint => <article className="checkpoint-card" key={checkpoint.id}
      aria-labelledby={'checkpoint-' + checkpoint.id}>
      <div className="checkpoint-image">
        {checkpoint.thumbnail_url
          ? <img src={checkpoint.thumbnail_url} alt={'Preparation: ' + checkpoint.name} loading="lazy" />
          : <div className="checkpoint-image-empty"><Camera size={30} aria-hidden="true" /><span>No preparation image</span></div>}
        <span className="checkpoint-image-label">Preparation scene</span>
      </div>
      <div className="checkpoint-content">
        <div className="checkpoint-heading">
          <div>
            <h2 id={'checkpoint-' + checkpoint.id}>{checkpoint.name}</h2>
            <time dateTime={checkpoint.created_at} title={new Date(checkpoint.created_at).toLocaleString()}>
              Saved {new Date(checkpoint.created_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })}
            </time>
          </div>
          <span className={'checkpoint-status' + (checkpoint.visible ? ' is-visible' : '')}>
            {checkpoint.visible && <Check size={13} aria-hidden="true" />}{checkpoint.visible ? 'Available in Run' : 'Hidden from Run'}
          </span>
        </div>
        <p className="checkpoint-task">{checkpoint.task}</p>
        {checkpoint.notes && <details className="checkpoint-notes"><summary>Setup notes</summary><p>{checkpoint.notes}</p></details>}
        <div className="checkpoint-footer">
          <div className="checkpoint-primary-actions">
            <button className="primary-button" disabled={busy || !checkpoint.visible} onClick={() => onOpen(checkpoint.id, 'run')}
              title={checkpoint.visible ? 'Load a working copy without starting robot motion' : 'Show this plan in Run to load it'}>
              Load in Run<ArrowRight size={16} aria-hidden="true" />
            </button>
            <button disabled={busy} onClick={() => onOpen(checkpoint.id, 'admin')}>
              <Pencil size={15} aria-hidden="true" />Open in Prepare
            </button>
          </div>
          <div className="checkpoint-tools">
            <button className="checkpoint-rename" disabled={busy} onClick={() => {
              const name = window.prompt('Checkpoint name', checkpoint.name);
              if (name?.trim()) onUpdate(checkpoint, { name: name.trim() });
            }}>Rename</button>
            <button className="checkpoint-visibility" role="switch" aria-checked={checkpoint.visible}
              aria-label="Show in Run" disabled={busy} onClick={() => onUpdate(checkpoint, { visible: !checkpoint.visible })}>
              <span className="checkpoint-switch" aria-hidden="true" /><span>Show in Run</span>
            </button>
          </div>
        </div>
      </div>
    </article>)}</div>

    {!filtered.length && <section className="library-empty">
      <FolderOpen size={32} aria-hidden="true" />
      <h2>{checkpoints.length ? 'No matching plans' : 'Your first plan starts in Prepare'}</h2>
      <p>{checkpoints.length ? 'Try another search or change the visibility filter.' : 'Prepare and refine a task, then save a checkpoint to find it here.'}</p>
      {checkpoints.length > 0 && <button onClick={() => { setQuery(''); setVisibility('all'); }}>Clear filters</button>}
    </section>}
    {filtered.length > 0 && <p className="library-footnote">Loading opens an independent working copy. Your saved plan stays unchanged, and robot motion starts only when you choose Run.</p>}
  </main>;
}
