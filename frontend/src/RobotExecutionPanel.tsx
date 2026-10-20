import type { ReactNode } from 'react';
import type { ExecutionState } from './types';
import './execution.css';

export const executionActiveStates = new Set(['starting_server', 'preparing', 'preview', 'running', 'stopping']);
export type ManualOperation = 'reset_joints' | 'open_gripper' | 'close_gripper';
export const resetJointsConfirmation =
  'Move the selected robot to its recorded reset joint pose? Review the robot helper’s pose and motion settings and ensure the entire arm path is clear. This is a joint-space move, not a collision-checked path.';

export default function RobotExecutionPanel({ execution, workspaceId, busy, pending, scriptReady,
  onControl, onManualControl, error, children }: {
  execution: ExecutionState | null;
  workspaceId: string;
  busy: boolean;
  pending: boolean;
  scriptReady: boolean;
  onControl: (operation: 'run' | 'skip' | 'cancel') => void;
  onManualControl: (operation: ManualOperation) => void;
  error?: string;
  children?: ReactNode;
}) {
  const active = Boolean(execution && executionActiveStates.has(execution.status));
  const ownsRun = execution?.session_id === workspaceId;
  const operationLabel = execution?.operation === 'reset_joints' ? 'Reset Joints'
    : execution?.operation === 'open_gripper' ? 'Open Gripper'
    : execution?.operation === 'close_gripper' ? 'Close Gripper' : 'Robot';
  return <section className="execution-panel script-panel-footer" aria-label="Robot execution">
    <div className="execution-actions">
      <strong>{operationLabel} · {execution?.status || 'idle'}</strong>
      <button className="primary-button" disabled={busy || !execution || !scriptReady} onClick={() => onControl('run')}>Run</button>
      <button disabled={pending || !ownsRun || execution?.status !== 'running' || !execution.subtask || execution.skip_pending} onClick={() => onControl('skip')}>Skip Subtask</button>
      <button className="cancel-button" disabled={!active || !ownsRun || execution?.status === 'stopping'} onClick={() => onControl('cancel')}>Cancel</button>
      <button disabled={busy || !execution} onClick={() => onManualControl('reset_joints')}>Reset Joints</button>
      <button disabled={busy || !execution} onClick={() => onManualControl('open_gripper')}>Open Gripper</button>
      <button disabled={busy || !execution} onClick={() => onManualControl('close_gripper')}>Close Gripper</button>
    </div>
    {execution?.status === 'preview' && <p role="status">Review 3D preview—close the window to continue</p>}
    {execution?.status === 'starting_server' && <p>Waiting for robot and gripper readiness…</p>}
    {execution?.status === 'stopping' && <p>Stopping owned processes and verifying hardware cleanup…</p>}
    {(error || execution?.error) && <p className="error-strip" role="alert">{error || execution?.error}</p>}
    <ol className="subtask-status">{execution?.subtasks.map(task => <li key={task.id}>
      <strong>Subtask {task.id}</strong> · {task.description} <span>{task.status}</span>
    </li>)}</ol>
    <details><summary>Robot execution logs</summary><pre className="execution-logs">{execution?.logs.map(line => '[' + line.role + '] ' + line.text).join('\n')}</pre></details>
    {children}
  </section>;
}
