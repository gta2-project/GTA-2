import { useEffect, useState } from 'react';
import { fetchJson, WorkspaceDashboard } from './App';
import CheckpointLibrary from './CheckpointLibrary';
import RobotExecutionPanel, { executionActiveStates, resetJointsConfirmation } from './RobotExecutionPanel';
import type { Checkpoint, LibrarySession, ExecutionState, WorkflowState } from './types';

export default function LibraryShell() {
  const page = window.location.pathname.replace(/\/$/, '') || '/admin';
  const runMode = page === '/run';
  const library = page === '/library';
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);
  const [session, setSession] = useState<LibrarySession | null>(null);
  const [selected, setSelected] = useState('');
  const [loaded, setLoaded] = useState(false);
  const [workflow, setWorkflow] = useState<WorkflowState | null>(null);
  const [execution, setExecution] = useState<ExecutionState | null>(null);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [pending, setPending] = useState(false);
  const [name, setName] = useState('');
  const [notes, setNotes] = useState('');
  const [visible, setVisible] = useState(true);
  const active = Boolean(execution && executionActiveStates.has(execution.status));
  const busy = pending || active || Boolean(workflow?.busy_stage || workflow?.active_handlers);
  const prefix = runMode && session ? '/api/library/sessions/' + session.id : '/api';
  const refreshLibrary = () => fetchJson<Checkpoint[]>('/api/checkpoints?visible_only=' + runMode).then(setCheckpoints);
  const refreshExecution = () => fetchJson<ExecutionState>('/api/execution').then(setExecution);

  useEffect(() => {
    Promise.all([refreshLibrary(), refreshExecution(), runMode
      ? fetchJson<LibrarySession | null>('/api/library/session').then(value => { setSession(value); setSelected(value?.checkpoint_id || ''); })
      : Promise.resolve()]).catch(e => setError(e.message)).finally(() => setLoaded(true));
    const timer = window.setInterval(() => { refreshExecution().catch(() => undefined); }, 1000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!runMode || !session) return;
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const socket = new WebSocket(protocol + '://' + window.location.host + prefix + '/events');
    socket.onmessage = message => {
      if (JSON.parse(message.data).type === 'execution') refreshExecution().catch(() => undefined);
    };
    return () => socket.close();
  }, [session?.id]);

  const action = async (fn: () => Promise<unknown>) => {
    setPending(true); setError(''); setNotice('');
    try { await fn(); } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setPending(false); refreshExecution().catch(() => undefined); }
  };

  const open = (checkpointId: string, target: 'admin' | 'run') => action(async () => {
    const next = await fetchJson<LibrarySession>('/api/library/sessions', {
      method: 'POST', body: JSON.stringify({ checkpoint_id: checkpointId, target })
    });
    if (runMode && target === 'run') {
      setSession(next); setWorkflow(null);
    } else {
      window.location.assign('/' + target);
    }
  });

  const save = () => action(async () => {
    await fetchJson('/api/checkpoints', { method: 'POST', body: JSON.stringify({ name, notes, visible, source: 'admin' }) });
    setNotice('Saved “' + name + '” to the Library.');
    setName(''); setNotes(''); await refreshLibrary();
  });

  const update = (checkpoint: Checkpoint, changes: Partial<Checkpoint>) => action(async () => {
    await fetchJson('/api/checkpoints/' + checkpoint.id, { method: 'PATCH', body: JSON.stringify(changes) });
    await refreshLibrary();
  });

  const control = (operation: 'run' | 'skip' | 'cancel') => action(async () => {
    setExecution(await fetchJson<ExecutionState>(prefix + '/execution/' + operation, {
      method: 'POST', body: JSON.stringify({ run_id: execution?.run_id, subtask: execution?.subtask })
    }));
  });

  const manualControl = (operation: 'reset_joints' | 'open_gripper' | 'close_gripper') => {
    if (operation === 'reset_joints' && !window.confirm(resetJointsConfirmation)) return;
    return action(async () => {
      setExecution(await fetchJson<ExecutionState>(prefix + '/execution/manual', {
        method: 'POST', body: JSON.stringify({ operation, confirmed: operation === 'reset_joints' })
      }));
    });
  };
  const savePanel = <section className="checkpoint-save script-panel-footer" aria-label="Save Checkpoint">
    <h3>Save Checkpoint</h3>
    <p>Save the complete plan, agent context, images and feedback to the Library.</p>
    <form onSubmit={event => { event.preventDefault(); save(); }}>
      <label>Name<input value={name} onChange={event => setName(event.target.value)} required disabled={busy} /></label>
      <label>Setup notes<textarea value={notes} onChange={event => setNotes(event.target.value)} disabled={busy} /></label>
      <label className="checkbox-label"><input type="checkbox" checked={visible} onChange={event => setVisible(event.target.checked)} disabled={busy} />Show in Run</label>
      <button disabled={busy || !name.trim() || !workflow?.parameters.length}>Save Checkpoint</button>
    </form>
    {notice && <p role="status">{notice}</p>}
  </section>;

  const runPanel = session && <RobotExecutionPanel execution={execution} workspaceId={session.id}
    busy={busy} pending={pending} scriptReady={Boolean(workflow?.script_ready)}
    onControl={control} onManualControl={manualControl}>
    <button disabled={busy} onClick={() => action(async () => {
      const next = await fetchJson<LibrarySession>(prefix + '/reset', { method: 'POST' });
      setSession(next); setWorkflow(null);
    })}>Reset to Checkpoint</button>
  </RobotExecutionPanel>;

  return <div className="library-app">
    <div className="library-toolbar">
      <nav className="library-tabs" aria-label="Library pages">
        <a href="/admin" aria-current={!runMode && !library ? 'page' : undefined}>Prepare</a>
        <a href="/library" aria-current={library ? 'page' : undefined}>Library</a>
        <a href="/run" aria-current={runMode ? 'page' : undefined}>Run</a>
      </nav>
      {loaded && runMode && <section className="library-selection" aria-label="Load checkpoint">
        <label htmlFor="library-checkpoint">Prepared checkpoint</label>
        <select id="library-checkpoint" value={selected} onChange={event => setSelected(event.target.value)} disabled={busy}>
          <option value="">Choose a checkpoint</option>
          {checkpoints.map(checkpoint => <option key={checkpoint.id} value={checkpoint.id}>{checkpoint.name}</option>)}
        </select>
        <button disabled={busy || !selected} onClick={() => open(selected, 'run')}>Load</button>
      </section>}
    </div>
    {error && <div className="error-strip" role="alert">{error}</div>}
    {!loaded && <p className="library-container">Loading…</p>}
    {loaded && library && <CheckpointLibrary checkpoints={checkpoints} busy={busy} onOpen={open} onUpdate={update} />}
    {loaded && runMode && !session && <p className="library-selection-hint">Choose a prepared plan here or from the Library. Loading never starts motion.</p>}
    {loaded && !library && (!runMode || session) && <WorkspaceDashboard
      key={prefix} apiPrefix={prefix} mode={runMode ? 'run' : 'admin'}
      executionActive={active || pending} onState={setWorkflow}
      scriptPanelFooter={runMode ? runPanel : savePanel}
    />}
  </div>;
}
