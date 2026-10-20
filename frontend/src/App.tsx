import {
  Activity,
  CheckCircle2,
  Circle,
  Cpu,
  Download,
  FileCode2,
  Image as ImageIcon,
  Layers3,
  Loader2,
  Play,
  RefreshCcw,
  Save,
  Send,
  Server,
  SlidersHorizontal,
  Waypoints,
  X
} from 'lucide-react';
import { PrismLight as SyntaxHighlighter } from 'react-syntax-highlighter';
import python from 'react-syntax-highlighter/dist/esm/languages/prism/python';
import { oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import type { FormEvent, ReactNode } from 'react';
import type { ArtifactInfo, EventItem, ExecutionState, Health, WorkflowState } from './types';
import RobotExecutionPanel, { executionActiveStates, resetJointsConfirmation } from './RobotExecutionPanel';
import type { ManualOperation } from './RobotExecutionPanel';

type JsonRecord = Record<string, unknown>;
type StageCompletionState = {
  task: boolean;
  skill: boolean;
  parameter: boolean;
  vision: boolean;
  script: boolean;
};

SyntaxHighlighter.registerLanguage('python', python);

function emptyState(): WorkflowState {
  return {
    task: null,
    subtasks: [],
    skills: [],
    parameters: [],
    keypoints: {},
    messages: [],
    events: [],
    artifacts: {},
    busy_stage: null,
    last_error: null,
    script: null,
    script_path: null,
    revision: 0,
    grounding_ready: false,
    script_ready: false,
    active_handlers: 0,
    image_revision: null
  };
}

export async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    headers: {
      'Content-Type': 'application/json'
    },
    ...options
  });
  const text = await response.text();
  let data: Record<string, unknown> = {};
  if (text) {
    try {
      data = JSON.parse(text) as Record<string, unknown>;
    } catch {
      if (!response.ok) {
        throw new Error(`Request failed (${response.status})`);
      }
      throw new Error('The server returned an invalid response');
    }
  }
  if (!response.ok) {
    throw new Error(typeof data.detail === 'string' ? data.detail : response.statusText);
  }
  return data as T;
}

export function WorkspaceDashboard({ apiPrefix = '/api', mode = 'standalone', executionActive = false,
  onState, scriptPanelFooter }: {
    apiPrefix?: string; mode?: 'standalone' | 'admin' | 'run'; executionActive?: boolean;
    onState?: (state: WorkflowState) => void; scriptPanelFooter?: ReactNode;
  }) {
  const runMode = mode === 'run';
  const managed = mode !== 'standalone';
  const requestJson = <T,>(url: string, options?: RequestInit) => fetchJson<T>(url.replace(/^\/api/, apiPrefix), options);
  const stageCompletionRef = useRef<StageCompletionState | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [state, setState] = useState<WorkflowState>(emptyState);
  const [task, setTask] = useState('');
  const [agentDrafts, setAgentDrafts] = useState<Record<string, string>>({
    TaskDecomposer: '',
    SkillGenerator: '',
    ParameterSetter: '',
    VisionModule: ''
  });
  const [scriptText, setScriptText] = useState('');
  const [actionError, setActionError] = useState<string | null>(null);
  const [savedWorkflowFolder, setSavedWorkflowFolder] = useState<string | null>(null);
  const [eventStreamConnected, setEventStreamConnected] = useState(false);

  const busy = Boolean(state.busy_stage || state.active_handlers || executionActive);
  const environmentImage = state.artifacts['environment.jpg'];
  const annotatedImage = state.artifacts['annotated_keypoints.png'];
  const generatedScript = state.artifacts['robot_script.py'];
  const hasGeneratedScript = managed ? state.script_ready : Boolean(state.script_path || state.script);
  const stageCompletionState: StageCompletionState = {
    task: state.subtasks.length > 0,
    skill: state.skills.length > 0,
    parameter: state.parameters.length > 0,
    vision: managed ? state.grounding_ready : Object.keys(state.keypoints || {}).length > 0,
    script: hasGeneratedScript
  };
  const autoAdvanceStageIds: Array<keyof StageCompletionState> = ['skill', 'parameter', 'vision'];

  const refreshState = async () => {
    const nextState = await requestJson<WorkflowState>('/api/state');
    setState(nextState);
    if (managed || nextState.script) setScriptText(nextState.script || '');
    onState?.(nextState);
  };

  const refreshHealth = async () => {
    setHealth(await requestJson<Health>('/api/health'));
  };

  useEffect(() => {
    refreshHealth().catch((error) => setActionError(error.message));
    refreshState().catch((error) => setActionError(error.message));

    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const socket = new WebSocket(`${protocol}://${window.location.host}${apiPrefix}/events`);
    socket.onopen = () => setEventStreamConnected(true);
    socket.onmessage = (message) => {
      setEventStreamConnected(true);
      const event = JSON.parse(message.data) as EventItem;
      if (event.type === 'execution') return;
      setState((current) => ({
        ...current,
        events: [...current.events, event].slice(-100)
      }));
      refreshState().catch((error) => setActionError(error.message));
    };
    socket.onerror = () => setEventStreamConnected(false);
    socket.onclose = () => setEventStreamConnected(false);

    const interval = window.setInterval(() => {
      refreshHealth().catch(() => undefined);
      refreshState().catch(() => undefined);
    }, 5000);

    return () => {
      window.clearInterval(interval);
      socket.close();
    };
  }, [apiPrefix]);

  const statusText = useMemo(() => {
    if (!health) {
      return 'Connecting';
    }
    if (!health.ready) {
      return 'Config needs attention';
    }
    return state.busy_stage || 'Ready';
  }, [health, state.busy_stage]);

  const middlewareStatus = useMemo(() => {
    if (!health) {
      return 'Checking middleware';
    }
    if (health.gta_backend === 'ros2') {
      return health.robot_dependencies.rclpy ? 'ROS 2 available' : 'ROS 2 unavailable';
    }
    if (health.gta_backend === 'ros1') {
      return health.robot_dependencies.rospy ? 'ROS available' : 'ROS unavailable';
    }
    return 'Unknown GTA backend';
  }, [health]);

  const startWorkflow = async (event: FormEvent) => {
    event.preventDefault();
    setActionError(null);
    setSavedWorkflowFolder(null);
    await requestJson('/api/workflow/start', {
      method: 'POST',
      body: JSON.stringify({ task })
    })
      .then(() => {
        stageCompletionRef.current = {
          task: false,
          skill: false,
          parameter: false,
          vision: false,
          script: false
        };
        setActiveStageId('task');
      })
      .catch((error) => setActionError(error.message));
    await refreshState().catch((error) => setActionError(error.message));
  };

  const sendAgentMessage = async (event: FormEvent, agentKey: string) => {
    event.preventDefault();
    const draft = agentDrafts[agentKey] || '';
    if (!draft.trim()) {
      return;
    }

    setActionError(null);
    await requestJson(`/api/agents/${agentKey}/message`, {
      method: 'POST',
      body: JSON.stringify({ message: draft })
    })
      .then(() =>
        setAgentDrafts((current) => ({
          ...current,
          [agentKey]: ''
        }))
      )
      .catch((error) => setActionError(error.message));
    await refreshState().catch((error) => setActionError(error.message));
  };

  const requeryVision = async () => {
    setActionError(null);
    await requestJson('/api/vision/requery', { method: 'POST' }).catch((error) =>
      setActionError(error.message)
    );
    await refreshState().catch((error) => setActionError(error.message));
  };

  const generateScript = async () => {
    setActionError(null);
    setSavedWorkflowFolder(null);
    await requestJson<{ script: string }>('/api/script/generate', { method: 'POST' })
      .then((result) => {
        setScriptText(result.script);
        if (managed) setActiveStageId('script');
      })
      .catch((error) => setActionError(error.message));
    await refreshState().catch((error) => setActionError(error.message));
  };

  const saveWorkflow = async () => {
    setActionError(null);
    setSavedWorkflowFolder(null);
    await requestJson<{ folder: string }>('/api/workflow/save', { method: 'POST' })
      .then((result) => setSavedWorkflowFolder(result.folder))
      .catch((error) => setActionError(error.message));
    await refreshState().catch((error) => setActionError(error.message));
  };

  const stagePanels = [
    {
      id: 'task',
      label: 'Task Decomposer',
      meta: `${state.subtasks.length} subtasks`,
      agentKey: 'TaskDecomposer',
      done: stageCompletionState.task,
      running: includesStage(state.busy_stage, 'Decomposing'),
      panel: (
        <TaskPlanPanel
          subtasks={state.subtasks}
          modelLabel={formatModelLabel(health?.agent_models?.TaskDecomposer)}
        />
      )
    },
    {
      id: 'skill',
      label: 'Skill Generator',
      meta: `${countControllers(state.skills)} controllers`,
      agentKey: 'SkillGenerator',
      done: stageCompletionState.skill,
      running: includesStage(state.busy_stage, 'skills'),
      panel: (
        <SkillRecipePanel
          skills={state.skills}
          modelLabel={formatModelLabel(health?.agent_models?.SkillGenerator)}
        />
      )
    },
    {
      id: 'parameter',
      label: 'Parameter Setter',
      meta: `${countOffsetParameters(state.parameters)} values`,
      agentKey: 'ParameterSetter',
      done: stageCompletionState.parameter,
      running: includesStage(state.busy_stage, 'parameters'),
      panel: (
        <ParameterPanel
          parameters={state.parameters}
          modelLabel={formatModelLabel(health?.agent_models?.ParameterSetter)}
        />
      )
    },
    {
      id: 'vision',
      label: 'Vision Module',
      meta: `${Object.keys(state.keypoints || {}).length} keypoints`,
      agentKey: 'VisionModule',
      done: stageCompletionState.vision,
      running: includesStage(state.busy_stage, 'keypoints') || includesStage(state.busy_stage, 'vision'),
      panel: (
        <VisionPanel
          parameters={state.parameters}
          keypoints={state.keypoints}
          busy={busy}
          onRequery={requeryVision}
          modelLabel={formatModelLabel(health?.agent_models?.VisionModule)}
        />
      )
    },
    {
      id: 'script',
      label: 'Robot Script',
      meta: hasGeneratedScript ? 'ready' : 'not generated',
      agentKey: null,
      done: stageCompletionState.script,
      running: false,
      panel: (
        <ArtifactPanel
          scriptText={scriptText}
          generatedScript={generatedScript}
          hasGeneratedScript={hasGeneratedScript}
          busy={busy}
          groundingReady={!managed || state.grounding_ready}
          onGenerateScript={generateScript}
          onSaveWorkflow={saveWorkflow}
          savedWorkflowFolder={savedWorkflowFolder}
          footer={scriptPanelFooter}
        />
      )
    }
  ];
  const [activeStageId, setActiveStageId] = useState(runMode ? 'vision' : stagePanels[0].id);

  const navigateToStage = (nextStageId: string) => {
    if (nextStageId === activeStageId) {
      return;
    }
    setActiveStageId(nextStageId);
  };

  useEffect(() => {
    const previousCompletionState = stageCompletionRef.current;
    if (!previousCompletionState) {
      stageCompletionRef.current = stageCompletionState;
      return;
    }

    const nextAutoStageId = autoAdvanceStageIds.find(
      (stageId) => stageCompletionState[stageId] && !previousCompletionState[stageId]
    );

    stageCompletionRef.current = stageCompletionState;

    if (nextAutoStageId) {
      setActiveStageId(nextAutoStageId);
    }
  }, [
    stageCompletionState.parameter,
    stageCompletionState.script,
    stageCompletionState.skill,
    stageCompletionState.task,
    stageCompletionState.vision
  ]);

  const activeStageIndex = stagePanels.findIndex((stage) => stage.id === activeStageId);
  const completedStageCount = stagePanels.filter((stage) => stage.done).length;
  const allStagesComplete = completedStageCount === stagePanels.length;
  const activeStageLabel = stagePanels[Math.max(activeStageIndex, 0)]?.label || 'Workflow';
  const scenePanel = <ScenePanel
    annotatedImage={!managed || state.grounding_ready ? annotatedImage : undefined}
    busyStage={state.busy_stage}
    cameraBackend={health?.camera_backend || 'camera'}
    environmentImage={environmentImage}
    keypoints={!managed || state.grounding_ready ? state.keypoints : {}}
  />;

  return (
    <main className="app-shell">
      {!managed && <header className="topbar">
        <div className="topbar-brand">
          <div className="brand-mark" aria-hidden="true">
            <Cpu size={19} />
          </div>
          <div className="topbar-copy">
            <h1>GTA Sessions</h1>
            <p>{state.task || 'Robot workflow console'}</p>
          </div>
        </div>
        <div className="status-cluster">
          <StatusPill
            icon={allStagesComplete ? <CheckCircle2 size={16} /> : <Circle size={16} />}
            label={`${completedStageCount}/${stagePanels.length} stages`}
            active={allStagesComplete}
          />
          <StatusPill icon={<Server size={16} />} label={statusText} active={Boolean(health?.ready)} />
          <StatusPill icon={<ImageIcon size={16} />} label={health?.camera_backend || 'camera'} />
          <StatusPill
            icon={<Activity size={16} />}
            label={middlewareStatus}
          />
        </div>
      </header>}

      {(actionError || state.last_error || health?.init_error) && (
        <section className="error-strip">
          {actionError || state.last_error || health?.init_error}
        </section>
      )}

      <section className="workspace">
        {!runMode && <section className="task-command-panel">
          <form className="task-command-form" onSubmit={startWorkflow}>
            <label htmlFor="task">Task</label>
            <div className="task-command-control">
              <input
                id="task"
                value={task}
                onChange={(event) => setTask(event.target.value)}
                placeholder="Describe the manipulation task"
                disabled={busy}
              />
              <button className="primary-button" type="submit" disabled={busy || !task.trim()}>
                {busy ? <Loader2 size={18} className="spin" /> : <Play size={18} />}
                Start Workflow
              </button>
            </div>
            <small>{busy ? state.busy_stage : 'Standing by'}</small>
          </form>
        </section>}
        <StageNavigator
          activeStageId={activeStageId}
          activeStageLabel={activeStageLabel}
          completedStageCount={completedStageCount}
          onNavigate={navigateToStage}
          stages={stagePanels}
        />

        <section className="main-layout">
          {scenePanel}

          <section className="detail-stack">
            <section className="stage-browser">
              <div className="stage-browser-current">
                <div
                  className="stage-browser-track"
                  style={{ transform: `translateX(-${Math.max(activeStageIndex, 0) * 100}%)` }}
                >
                  {stagePanels.map((stage) => (
                    <div className="stage-browser-slide" key={stage.id}>
                      <div className="stage-browser-slide-content">
                        {stage.panel}
                        {stage.agentKey && <RefinementGroup runMode={runMode} vision={stage.agentKey === 'VisionModule'} title={stage.label}>
                          <AgentRefinePanel
                            agentKey={stage.agentKey}
                            disabled={busy || (managed && stage.agentKey === 'VisionModule' && !state.grounding_ready)}
                            message={agentDrafts[stage.agentKey] || ''}
                            onChange={(value) =>
                              setAgentDrafts((current) => ({
                                ...current,
                                [stage.agentKey as string]: value
                              }))
                            }
                            onSubmit={sendAgentMessage}
                            title={stage.label}
                          />
                        </RefinementGroup>}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </section>
            <EventStream connected={eventStreamConnected} events={state.events.filter(event => event.type !== 'execution')} />
          </section>
        </section>
      </section>
    </main>
  );
}

function RefinementGroup({ runMode, vision, title, children }: {
  runMode: boolean; vision: boolean; title: string; children: ReactNode;
}) {
  if (!runMode) return <>{children}</>;
  return <details className="refinement-panel" open={vision}>
    <summary>{vision ? 'Vision feedback on this captured image' : `Refine ${title}`}</summary>
    {children}
  </details>;
}

function StageNavigator({
  stages,
  activeStageId,
  activeStageLabel,
  completedStageCount,
  onNavigate
}: {
  stages: Array<{ id: string; label: string; meta: string; done: boolean; running: boolean }>;
  activeStageId: string;
  activeStageLabel: string;
  completedStageCount: number;
  onNavigate: (stageId: string) => void;
}) {
  return (
    <section className="stage-browser-nav">
      <div className="stage-browser-meta">
        <span>Workflow</span>
        <strong>{activeStageLabel}</strong>
        <small>{completedStageCount}/{stages.length}</small>
      </div>
      <div className="workflow-progress" aria-hidden="true">
        <span style={{ width: `${(completedStageCount / stages.length) * 100}%` }} />
      </div>
      <div
        className="stage-browser-tabs"
        style={{ gridTemplateColumns: `repeat(${stages.length}, minmax(0, 1fr))` }}
      >
        {stages.map((stage, index) => {
          const statusLabel = stage.running ? 'Running' : stage.done ? 'Ready' : 'Queued';

          return (
            <button
              aria-current={stage.id === activeStageId ? 'step' : undefined}
              className={`stage-tab ${stage.id === activeStageId ? 'active' : ''} ${stage.done ? 'done' : ''} ${stage.running ? 'running' : ''}`}
              key={stage.id}
              onClick={() => onNavigate(stage.id)}
              type="button"
            >
              <span className="stage-tab-leading">
                <span className="stage-tab-index">{String(index + 1).padStart(2, '0')}</span>
                <span className="stage-tab-copy">
                  <span className="stage-tab-label">{stage.label}</span>
                  <span className="stage-tab-meta">{stage.meta}</span>
                </span>
              </span>
              <span className="stage-tab-state" aria-label={statusLabel} title={statusLabel}>
                {stage.running ? (
                  <Loader2 size={14} className="spin" />
                ) : stage.done ? (
                  <CheckCircle2 size={14} />
                ) : (
                  <Circle size={14} />
                )}
              </span>
            </button>
          );
        })}
      </div>
    </section>
  );
}

function ScenePanel({
  environmentImage,
  annotatedImage,
  keypoints,
  cameraBackend,
  busyStage
}: {
  environmentImage?: ArtifactInfo;
  annotatedImage?: ArtifactInfo;
  keypoints: Record<string, unknown>;
  cameraBackend: string;
  busyStage: string | null;
}) {
  const groups = groupKeypoints(keypoints);
  const keypointGroups = Object.entries(groups);
  const keypointCount = keypointGroups.reduce((total, [, items]) => total + items.length, 0);
  const environmentMtime = environmentImage?.mtime ? Date.parse(environmentImage.mtime) : null;
  const annotatedMtime = annotatedImage?.mtime ? Date.parse(annotatedImage.mtime) : null;
  const annotationIsCurrent =
    Boolean(annotatedImage?.exists) &&
    keypointCount > 0 &&
    (environmentMtime === null ||
      (!Number.isNaN(environmentMtime) &&
        annotatedMtime !== null &&
        !Number.isNaN(annotatedMtime) &&
        annotatedMtime >= environmentMtime));
  const displayImage = annotationIsCurrent ? annotatedImage : environmentImage?.exists ? environmentImage : undefined;
  const imageMode = annotationIsCurrent ? 'Annotated keypoints' : 'Environment image';
  const imageSrc = displayImage?.url
    ? `${displayImage.url}?v=${encodeURIComponent(displayImage.mtime || '')}`
    : null;
  const [isPreviewOpen, setIsPreviewOpen] = useState(false);

  useEffect(() => {
    if (!imageSrc) {
      setIsPreviewOpen(false);
    }
  }, [imageSrc]);

  useEffect(() => {
    if (!isPreviewOpen) {
      return;
    }

    const previousOverflow = document.body.style.overflow;
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setIsPreviewOpen(false);
      }
    };

    document.body.style.overflow = 'hidden';
    window.addEventListener('keydown', closeOnEscape);

    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener('keydown', closeOnEscape);
    };
  }, [isPreviewOpen]);

  return (
    <aside className="scene-panel domain-panel">
      <div className="panel-heading">
        <div>
          <h2>
            <ImageIcon size={19} />
            Scene
          </h2>
          <span>{imageMode}</span>
        </div>
        <Badge>{cameraBackend}</Badge>
      </div>
      <div className="panel-content scene-content">
        <div className="scene-frame">
          {imageSrc ? (
            <button
              aria-label={`Open ${imageMode} preview`}
              className="scene-image-button"
              onClick={() => setIsPreviewOpen(true)}
              type="button"
            >
              <img src={imageSrc} alt={imageMode} />
            </button>
          ) : (
            <div className="empty-state">No environment image</div>
          )}
        </div>

        <div className="scene-facts">
          <div>
            <span>Vision</span>
            <strong>{busyStage && includesStage(busyStage, 'vision') ? busyStage : imageMode}</strong>
          </div>
          <div>
            <span>Keypoints</span>
            <strong>{keypointCount}</strong>
          </div>
        </div>

        {keypointGroups.length > 0 && (
          <div className="scene-keypoints">
            <span>Detected keypoints</span>
            <div className="scene-keypoint-groups">
              {keypointGroups.map(([objectName, items]) => (
                <div className="scene-keypoint-row" key={objectName}>
                  <strong>{objectName}</strong>
                  <div>
                    {items.map((item) => (
                      <Badge key={item.name}>{item.partLabel}</Badge>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {imageSrc && isPreviewOpen &&
        createPortal(
          <div
            aria-label={imageMode}
            aria-modal="true"
            className="image-lightbox"
            onClick={() => setIsPreviewOpen(false)}
            role="dialog"
          >
            <button
              aria-label="Close image preview"
              className="image-lightbox-dismiss"
              onClick={() => setIsPreviewOpen(false)}
              type="button"
            >
              <X size={18} />
            </button>
            <div className="image-lightbox-shell" onClick={(event) => event.stopPropagation()}>
              <img src={imageSrc} alt={imageMode} />
            </div>
          </div>,
          document.body
        )}
    </aside>
  );
}

function AgentRefinePanel({
  agentKey,
  title,
  message,
  disabled,
  onChange,
  onSubmit
}: {
  agentKey: string;
  title: string;
  message: string;
  disabled: boolean;
  onChange: (value: string) => void;
  onSubmit: (event: FormEvent, agentKey: string) => Promise<void>;
}) {
  return (
    <section className="domain-panel agent-refine-panel">
      <div className="panel-heading">
        <div>
          <h2>
            <Send size={18} />
            Feedback
          </h2>
          <span>{title}</span>
        </div>
      </div>
      <div className="panel-content">
        <form className="agent-form" onSubmit={(event) => void onSubmit(event, agentKey)}>
          <textarea
            rows={1}
            value={message}
            onChange={(event) => onChange(event.target.value)}
            placeholder="Specific feedback, correction, or refinement"
            disabled={disabled}
          />
          <button type="submit" disabled={disabled || !message.trim()}>
            <Send size={17} />
            Send
          </button>
        </form>
      </div>
    </section>
  );
}

function TaskPlanPanel({ subtasks, modelLabel }: { subtasks: JsonRecord[]; modelLabel: string }) {
  return (
    <Panel
      title="Task Decomposer"
      icon={<Waypoints size={19} />}
      meta={`${subtasks.length} subtasks created`}
      empty={subtasks.length === 0}
      emptyText="Awaiting task plan."
      action={<AgentHeaderAction modelLabel={modelLabel} />}
    >
      <div className="timeline">
        {subtasks.map((subtask, index) => (
          <article className="phase-card compact" key={getString(subtask, 'id', String(index + 1))}>
            <div className="phase-index">{getString(subtask, 'id', String(index + 1))}</div>
            <div className="phase-body">
              <p className="phase-task">
                {getString(subtask, 'subtask_prompt', getString(subtask, 'description', 'Untitled phase'))}
              </p>
              <p className="phase-goal">
                <span>Goal check</span>
                {getString(subtask, 'goal_check_prompt', 'No goal check supplied')}
              </p>
            </div>
          </article>
        ))}
      </div>
      <InspectData data={subtasks} />
    </Panel>
  );
}

function SkillRecipePanel({ skills, modelLabel }: { skills: JsonRecord[]; modelLabel: string }) {
  const references = collectSkillReferences(skills);

  return (
    <Panel
      title="Skill Generator"
      icon={<Layers3 size={19} />}
      meta={`${skills.length} recipes, ${countControllers(skills)} controllers`}
      empty={skills.length === 0}
      emptyText="Awaiting skill recipes."
      action={<AgentHeaderAction modelLabel={modelLabel} />}
    >
      <SkillReferenceSummary references={references} />
      <div className="recipe-list">
        {skills.map((skill, index) => {
          const controllers = asRecordArray(skill.controllers);
          return (
            <article
              className="phase-card compact recipe-compact"
              key={`${getString(skill, 'subtask_id', String(index + 1))}-${index}`}
            >
              <div className="phase-index">{getString(skill, 'subtask_id', String(index + 1))}</div>
              <div className="phase-body">
                <div className="recipe-summary">
                  <p className="phase-task">{getString(skill, 'task', 'Untitled skill')}</p>
                  <Badge>{controllers.length} controllers</Badge>
                </div>

                <div className="controller-stack compact">
                  {controllers.length === 0 && <div className="empty-inline">No controllers</div>}
                  {controllers.map((controller, controllerIndex) => (
                    <ControllerCard
                      controller={controller}
                      key={`${getString(controller, 'controller', 'controller')}-${controllerIndex}`}
                    />
                  ))}
                </div>
              </div>
            </article>
          );
        })}
      </div>
      <InspectData data={skills} />
    </Panel>
  );
}

function SkillReferenceSummary({
  references
}: {
  references: ReturnType<typeof collectSkillReferences>;
}) {
  const keypointGroups = Object.entries(references.keypoints);

  if (keypointGroups.length === 0) {
    return null;
  }

  return (
    <section className="skill-reference-summary">
      {keypointGroups.length > 0 && (
        <div className="reference-section">
          <span>Keypoints</span>
          <div className="reference-groups">
            {keypointGroups.map(([objectName, keypoints]) => (
              <div className="reference-row" key={objectName}>
                <strong>{objectName}</strong>
                <div>
                  {keypoints.map((keypoint) => (
                    <Badge key={`${objectName}-${keypoint}`}>{keypoint}</Badge>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}

function ParameterPanel({ parameters, modelLabel }: { parameters: JsonRecord[]; modelLabel: string }) {
  const totalOffsetValues = countOffsetParameters(parameters);
  const references = collectSkillReferences(parameters);

  return (
    <Panel
      title="Parameter Setter"
      icon={<SlidersHorizontal size={19} />}
      meta={`${parameters.length} recipes, ${totalOffsetValues} offset values`}
      empty={parameters.length === 0}
      emptyText="Awaiting parameter estimates."
      action={<AgentHeaderAction modelLabel={modelLabel} />}
    >
      <SkillReferenceSummary references={references} />
      <ParameterRecipeList parameters={parameters} emptyText="No offset parameters found in the current skills." />
      <InspectData data={parameters} />
    </Panel>
  );
}

function VisionPanel({
  parameters,
  keypoints,
  busy,
  onRequery,
  modelLabel
}: {
  parameters: JsonRecord[];
  keypoints: Record<string, unknown>;
  busy: boolean;
  onRequery: () => void;
  modelLabel: string;
}) {
  const groups = groupKeypoints(keypoints);
  const count = Object.values(groups).reduce((total, items) => total + items.length, 0);
  const keypointGroups = Object.entries(groups);

  return (
    <Panel
      title="Vision & Keypoints"
      icon={<ImageIcon size={19} />}
      meta={`${count} detected keypoints`}
      empty={count === 0}
      emptyText="Awaiting vision results."
      action={
        <AgentHeaderAction
          modelLabel={modelLabel}
          action={
            <button onClick={onRequery} disabled={busy}>
              <RefreshCcw size={17} />
              Requery
            </button>
          }
        />
      }
    >
      <div className="vision-dashboard">
        <section className="skill-reference-summary vision-reference-summary">
          <div className="reference-section vision-reference-section">
            <span>Keypoints</span>
            <div className="reference-groups">
              {keypointGroups.map(([objectName, items]) => (
                <div className="reference-row" key={objectName}>
                  <strong>{objectName}</strong>
                  <div>
                    {items.map((item) => (
                      <Badge key={item.name}>{`${item.partLabel}: ${item.point}`}</Badge>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>
      </div>
      <ParameterRecipeList parameters={parameters} emptyText="No parameter output carried from the previous agent yet." />
      <InspectData data={keypoints} />
    </Panel>
  );
}

function ParameterRecipeList({ parameters, emptyText }: { parameters: JsonRecord[]; emptyText: string }) {
  const totalOffsetValues = countOffsetParameters(parameters);

  if (totalOffsetValues === 0) {
    return <div className="empty-inline">{emptyText}</div>;
  }

  return (
    <div className="recipe-list">
      {parameters.map((skill, index) => {
        const controllers = asRecordArray(skill.controllers);
        return (
          <article
            className="phase-card compact recipe-compact"
            key={`${getString(skill, 'subtask_id', String(index + 1))}-${index}`}
          >
            <div className="phase-index">{getString(skill, 'subtask_id', String(index + 1))}</div>
            <div className="phase-body">
              <div className="recipe-summary">
                <p className="phase-task">{getString(skill, 'task', 'Untitled skill')}</p>
                <Badge>{countOffsetParameters([skill])} offsets</Badge>
              </div>

              <div className="controller-stack compact">
                {controllers.length === 0 && <div className="empty-inline">No controllers</div>}
                {controllers.map((controller, controllerIndex) => (
                  <ParameterControllerCard
                    controller={controller}
                    key={`${getString(controller, 'controller', 'controller')}-${controllerIndex}`}
                  />
                ))}
              </div>
            </div>
          </article>
        );
      })}
    </div>
  );
}

function ArtifactPanel({
  scriptText,
  generatedScript,
  hasGeneratedScript,
  busy,
  groundingReady,
  onGenerateScript,
  onSaveWorkflow,
  savedWorkflowFolder,
  footer
}: {
  scriptText: string;
  generatedScript?: ArtifactInfo;
  hasGeneratedScript: boolean;
  busy: boolean;
  groundingReady: boolean;
  onGenerateScript: () => void;
  onSaveWorkflow: () => void;
  savedWorkflowFolder: string | null;
  footer?: ReactNode;
}) {
  const scriptPreviewText =
    scriptText ||
    (hasGeneratedScript ? 'Script generated. Generate again to refresh this preview.' : 'No script generated.');

  return (
    <Panel
      title="Robot Script"
      icon={<FileCode2 size={19} />}
      meta={hasGeneratedScript ? 'robot_script.py ready' : 'Script pending'}
      action={
        <div className="script-action-row">
          <button className="primary-button" onClick={onGenerateScript} disabled={busy || !groundingReady}>
            <FileCode2 size={17} />
            Generate Script
          </button>
          {generatedScript?.url && (
            <a className="button-link" download="robot_script.py" href={generatedScript.url}>
              <Download size={17} />
              Download
            </a>
          )}
          <button onClick={onSaveWorkflow} disabled={busy}>
            <Save size={17} />
            Save Workflow
          </button>
        </div>
      }
    >
      {savedWorkflowFolder && (
        <div className="save-workflow-status">Saved to {savedWorkflowFolder}</div>
      )}
      <div className="script-preview">
        <SyntaxHighlighter
          language="python"
          style={oneLight}
          wrapLongLines
          customStyle={{
            margin: 0,
            minHeight: '100%',
            maxHeight: 'none',
            border: '0',
            borderRadius: 0,
            background: 'transparent',
            padding: '12px'
          }}
          codeTagProps={{
            style: {
              fontFamily: '"SFMono-Regular", Consolas, "Liberation Mono", monospace',
              fontSize: '0.78rem'
            }
          }}
        >
          {scriptPreviewText}
        </SyntaxHighlighter>
      </div>
      {footer}
    </Panel>
  );
}

function EventStream({ connected, events }: { connected: boolean; events: EventItem[] }) {
  return (
    <section className="event-panel">
      <details>
        <summary>
          <span>Event Stream</span>
          <small>{connected ? `${events.length} recent events` : `${events.length} recent events, polling`}</small>
        </summary>
        <div className="event-list">
          {events.length === 0 && <div className="empty-state">No events</div>}
          {events
            .slice()
            .reverse()
            .map((event, index) => (
              <article className={`event-row ${event.level}`} key={`${event.id}-${index}`}>
                <div>
                  <strong>{event.sender}</strong>
                  <span>{event.message_type}</span>
                </div>
                <code>{safePreview(event.content)}</code>
              </article>
            ))}
        </div>
      </details>
    </section>
  );
}

function Panel({
  title,
  icon,
  meta,
  children,
  empty = false,
  emptyText = 'Waiting',
  action
}: {
  title: string;
  icon: ReactNode;
  meta: string;
  children: ReactNode;
  empty?: boolean;
  emptyText?: string;
  action?: ReactNode;
}) {
  return (
    <section className="domain-panel">
      <div className="panel-heading">
        <div>
          <h2>
            {icon}
            {title}
          </h2>
          <span>{meta}</span>
        </div>
        {action}
      </div>
      <div className="panel-content">{empty ? <div className="empty-state">{emptyText}</div> : children}</div>
    </section>
  );
}

function AgentHeaderAction({ modelLabel, action }: { modelLabel: string; action?: ReactNode }) {
  return (
    <div className="panel-action">
      <small className="panel-model">{modelLabel}</small>
      {action}
    </div>
  );
}

function ControllerCard({ controller }: { controller: JsonRecord }) {
  const parsed = parseController(getString(controller, 'controller', ''));
  const theme = controllerTheme(parsed.name);

  return (
    <details className={`controller-card compact controller-details ${theme}`}>
      <summary className="controller-summary">
        <div className="controller-priority">
          <strong>{getString(controller, 'priority', '-')}</strong>
        </div>
        <div className="controller-main">
          <div className="controller-title">
            <Badge tone={theme}>{parsed.name || 'Controller'}</Badge>
            <p>{getString(controller, 'explanation', 'No explanation supplied')}</p>
          </div>
        </div>
      </summary>
      <code className="controller-code">{getString(controller, 'controller', '')}</code>
    </details>
  );
}

function ParameterControllerCard({ controller }: { controller: JsonRecord }) {
  const parsed = parseController(getString(controller, 'controller', ''));
  const theme = controllerTheme(parsed.name);
  const offsetParameters = asRecordArray(controller.offset_parameters);

  return (
    <details className={`controller-card compact controller-details parameter-controller ${theme}`}>
      <summary className="controller-summary">
        <div className="controller-priority">
          <strong>{getString(controller, 'priority', '-')}</strong>
        </div>
        <div className="controller-main">
          <div className="controller-title">
            <Badge tone={theme}>{parsed.name || 'Controller'}</Badge>
            <p>{getString(controller, 'explanation', 'No explanation supplied')}</p>
          </div>
          {offsetParameters.length > 0 && (
            <div className="parameter-tag-list">
              {offsetParameters.map((parameter, index) => (
                <Badge
                  key={`${getString(parameter, 'name', 'offset')}-${index}`}
                  tone={theme}
                >
                  {formatParameterSummary(parameter)}
                </Badge>
              ))}
            </div>
          )}
        </div>
      </summary>
      {offsetParameters.length > 0 && (
        <div className="parameter-reasoning-list">
          {offsetParameters.map((parameter, index) => (
            <section
              className="parameter-reasoning-item"
              key={`${getString(parameter, 'name', 'offset')}-reasoning-${index}`}
            >
              <div className="parameter-reasoning-header">
                <strong>{formatParameterSummary(parameter)}</strong>
              </div>
              <p>{getString(parameter, 'reasoning', getString(parameter, 'explanation', 'No reasoning supplied'))}</p>
            </section>
          ))}
        </div>
      )}
    </details>
  );
}

function KeypointChips({ keypoints }: { keypoints: JsonRecord[] }) {
  const objectKeypoints = keypoints.filter((keypoint) => {
    const parsed = parseKeypointName(getString(keypoint, 'name', ''));
    return Boolean(parsed.objectName);
  });

  if (objectKeypoints.length === 0) {
    return <div className="empty-inline">No keypoints listed</div>;
  }

  const groups = objectKeypoints.reduce<Record<string, JsonRecord[]>>((acc, keypoint) => {
    const parsed = parseKeypointName(getString(keypoint, 'name', ''));
    const objectName = parsed.objectName;
    acc[objectName] = [...(acc[objectName] || []), keypoint];
    return acc;
  }, {});

  return (
    <div className="keypoint-chip-groups">
      {Object.entries(groups).map(([objectName, items]) => (
        <div className="chip-group" key={objectName}>
          <span>{objectName}</span>
          <div>
            {items.map((keypoint, index) => {
              const parsed = parseKeypointName(getString(keypoint, 'name', ''));
              return (
                <Badge key={`${getString(keypoint, 'name', 'kp')}-${index}`}>
                  {parsed.partLabel || getString(keypoint, 'name', 'keypoint')}
                </Badge>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}

function RecoveryList({ recovery }: { recovery: JsonRecord[] }) {
  if (recovery.length === 0) {
    return null;
  }

  return (
    <details className="quiet-details">
      <summary>{recovery.length} recovery prompt{recovery.length === 1 ? '' : 's'}</summary>
      <div className="recovery-list">
        {recovery.map((item, index) => (
          <div className="recovery-item" key={index}>
            <InfoRow label="Detect" value={getString(item, 'detect_prompt', 'Not specified')} />
            <InfoRow label="Recover" value={getString(item, 'recovery_prompt', 'Not specified')} />
          </div>
        ))}
      </div>
    </details>
  );
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="info-row">
      <span>{label}</span>
      <p>{value}</p>
    </div>
  );
}

function StatusPill({
  icon,
  label,
  active = true
}: {
  icon: ReactNode;
  label: string;
  active?: boolean;
}) {
  return (
    <div className={`status-pill ${active ? 'active' : 'attention'}`}>
      {icon}
      <span>{label}</span>
    </div>
  );
}

function Badge({ children, tone = 'neutral' }: { children: ReactNode; tone?: string }) {
  return <span className={`badge ${tone}`}>{children}</span>;
}

function InspectData({ data }: { data: unknown }) {
  return (
    <details className="inspect-data">
      <summary>Inspect data</summary>
      <pre>{JSON.stringify(data, null, 2)}</pre>
    </details>
  );
}

function parseController(controller: string) {
  const match = controller.match(/^([A-Za-z0-9_]+)\((.*)\)$/);
  if (!match) {
    return { name: controller, args: [] };
  }
  return {
    name: match[1],
    args: splitArgs(match[2])
  };
}

function splitArgs(input: string) {
  const args: string[] = [];
  let depth = 0;
  let current = '';

  for (const char of input) {
    if (char === ',' && depth === 0) {
      args.push(current.trim());
      current = '';
      continue;
    }
    if (char === '(' || char === '[') {
      depth += 1;
    }
    if (char === ')' || char === ']') {
      depth -= 1;
    }
    current += char;
  }

  if (current.trim()) {
    args.push(current.trim());
  }
  return args;
}

function collectSkillReferences(skills: JsonRecord[]) {
  const keypointGroups: Record<string, Set<string>> = {};

  const addKeypoint = (name: string) => {
    const parsed = parseKeypointName(name);
    if (!parsed.objectName) {
      return;
    }
    if (!keypointGroups[parsed.objectName]) {
      keypointGroups[parsed.objectName] = new Set<string>();
    }
    keypointGroups[parsed.objectName].add(parsed.partLabel || name);
  };

  skills.forEach((skill) => {
    asRecordArray(skill.keypoints).forEach((keypoint) => {
      addKeypoint(getString(keypoint, 'name', ''));
    });

    asRecordArray(skill.controllers).forEach((controller) => {
      const controllerString = getString(controller, 'controller', '');
      controllerString.match(/g_[A-Za-z0-9_]+/g)?.forEach(addKeypoint);
    });
  });

  return {
    keypoints: Object.fromEntries(
      Object.entries(keypointGroups)
        .sort(([a], [b]) => a.localeCompare(b))
        .map(([objectName, keypoints]) => [objectName, Array.from(keypoints).sort()])
    )
  };
}

function parseKeypointName(name: string) {
  const parts = name.split('_');
  let objectName = '';
  let partName = '';
  let locationName = '';
  for (const part of parts) {
    if (part.startsWith('O')) {
      objectName = part.slice(1);
    }
    if (part.startsWith('P')) {
      partName = part.slice(1);
    }
    if (part.startsWith('L')) {
      locationName = part.slice(1);
    }
  }
  return {
    objectName,
    partName,
    locationName,
    partLabel: [partName, locationName].filter(Boolean).join(' ')
  };
}

function groupKeypoints(keypoints: Record<string, unknown>) {
  return Object.entries(keypoints).reduce<Record<string, Array<{ name: string; partLabel: string; query: string; point: string }>>>(
    (acc, [name, value]) => {
      const record = asRecord(value);
      const parsed = parseKeypointName(name);
      const objectName = parsed.objectName || 'Scene';
      const point = Array.isArray(record.point) ? `[${record.point.join(', ')}]` : '-';
      acc[objectName] = [
        ...(acc[objectName] || []),
        {
          name,
          partLabel: parsed.partLabel || name,
          query: getString(record, 'query', getString(record, 'label', name)),
          point
        }
      ].sort((left, right) => left.partLabel.localeCompare(right.partLabel));
      return acc;
    },
    {}
  );
}

function classifyParameter(parameter: JsonRecord) {
  const name = getString(parameter, 'name', '').toLowerCase();
  const unit = getString(parameter, 'unit', '').toLowerCase();
  if (unit.includes('n') || name.includes('force')) {
    return 'force';
  }
  if (unit.includes('deg') || unit.includes('rad') || name.includes('angle') || name.includes('rotation')) {
    return 'angle';
  }
  if (unit.includes('m') || name.includes('height') || name.includes('distance') || name.includes('offset')) {
    return 'distance';
  }
  return 'neutral';
}

function controllerTheme(name: string) {
  const themes: Record<string, string> = {
    PosAlign: 'controller-pos-align',
    AxisAlign: 'controller-axis-align',
    PosWaypoint: 'controller-pos-waypoint',
    AxisWaypoint: 'controller-axis-waypoint',
    ForceConstant: 'controller-force-constant',
    GripperControl: 'controller-gripper-control',
    PosDMP: 'controller-pos-dmp'
  };
  return themes[name] || 'controller-default';
}

function formatAgentLabel(agent: string) {
  const labels: Record<string, string> = {
    TaskDecomposer: 'Task Decomposer',
    SkillGenerator: 'Skill Generator',
    ParameterSetter: 'Parameter Setter',
    VisionModule: 'Vision Module'
  };
  return labels[agent] || agent;
}

function formatModelLabel(agentModel?: { provider: string | null; model: string | null }) {
  if (!agentModel?.model) {
    return 'model unavailable';
  }
  return agentModel.provider ? `${agentModel.provider}: ${agentModel.model}` : agentModel.model;
}

function countControllers(skills: JsonRecord[]) {
  return skills.reduce((count, skill) => count + asRecordArray(skill.controllers).length, 0);
}

function countOffsetParameters(skills: JsonRecord[]) {
  return skills.reduce(
    (count, skill) =>
      count +
      asRecordArray(skill.controllers).reduce(
        (controllerCount, controller) => controllerCount + asRecordArray(controller.offset_parameters).length,
        0
      ),
    0
  );
}

function formatParameterSummary(parameter: JsonRecord) {
  const name = getString(parameter, 'name', 'offset');
  const value = formatValue(parameter.value);
  const unit = getString(parameter, 'unit', '');
  return `${name}: ${value}${unit ? ` ${unit}` : ''}`;
}

function includesStage(stage: string | null, value: string) {
  return Boolean(stage?.toLowerCase().includes(value.toLowerCase()));
}

function getString(record: JsonRecord, key: string, fallback = '') {
  const value = record[key];
  if (value === null || value === undefined) {
    return fallback;
  }
  return String(value);
}

function asRecord(value: unknown): JsonRecord {
  return value && typeof value === 'object' && !Array.isArray(value) ? (value as JsonRecord) : {};
}

function asRecordArray(value: unknown): JsonRecord[] {
  return Array.isArray(value) ? value.map(asRecord) : [];
}

function asStringArray(value: unknown): string[] {
  return Array.isArray(value) ? value.map((item) => String(item)) : [];
}

function formatValue(value: unknown) {
  if (typeof value === 'number') {
    return Number.isInteger(value) ? String(value) : value.toFixed(3).replace(/0+$/, '').replace(/\.$/, '');
  }
  return value === null || value === undefined ? '-' : String(value);
}

function summarize(value: string, maxLength: number) {
  if (value.length <= maxLength) {
    return value;
  }
  return `${value.slice(0, maxLength - 1)}...`;
}

function safePreview(value: unknown) {
  const serialized = JSON.stringify(value);
  if (!serialized) {
    return '';
  }
  return serialized.length > 220 ? `${serialized.slice(0, 220)}...` : serialized;
}

export default function App() {
  const [workflow, setWorkflow] = useState<WorkflowState | null>(null);
  const [execution, setExecution] = useState<ExecutionState | null>(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState('');
  const [connectionError, setConnectionError] = useState('');
  const pendingRef = useRef(false);
  const requestVersion = useRef(0);
  const active = Boolean(execution && executionActiveStates.has(execution.status));
  const busy = pending || active || Boolean(workflow?.busy_stage || workflow?.active_handlers);

  useEffect(() => {
    let mounted = true;
    const refresh = async () => {
      if (pendingRef.current) return;
      const version = ++requestVersion.current;
      try {
        const next = await fetchJson<ExecutionState>('/api/execution');
        if (mounted && version === requestVersion.current) {
          setExecution(next);
          setConnectionError('');
        }
      } catch (e) {
        if (mounted && version === requestVersion.current) {
          setConnectionError(e instanceof Error ? e.message : String(e));
          setExecution(null);
        }
      }
    };
    refresh();
    const timer = window.setInterval(refresh, 1000);
    return () => { mounted = false; window.clearInterval(timer); };
  }, []);

  const action = async (operation: string, payload: object) => {
    // Cancel stays available during another request; duplicate non-cancel
    // clicks must not start a second manual operation after the first ends.
    if (pendingRef.current && operation !== 'cancel') return;
    const version = ++requestVersion.current;
    pendingRef.current = true;
    setPending(true);
    setError('');
    try {
      const next = await fetchJson<ExecutionState>('/api/execution/' + operation, {
        method: 'POST', body: JSON.stringify(payload)
      });
      if (version === requestVersion.current) setExecution(next);
    } catch (e) {
      if (version === requestVersion.current) setError(e instanceof Error ? e.message : String(e));
    } finally {
      if (version === requestVersion.current) {
        pendingRef.current = false;
        setPending(false);
      }
    }
  };
  const control = (operation: 'run' | 'skip' | 'cancel') =>
    action(operation, { run_id: execution?.run_id, subtask: execution?.subtask });
  const manualControl = (operation: ManualOperation) => {
    if (operation === 'reset_joints' && !window.confirm(resetJointsConfirmation)) return;
    return action('manual', { operation, confirmed: operation === 'reset_joints' });
  };

  return <WorkspaceDashboard executionActive={active || pending} onState={setWorkflow}
    scriptPanelFooter={<RobotExecutionPanel execution={execution} workspaceId="main"
      busy={busy} pending={pending} scriptReady={Boolean(workflow?.script_ready)} error={connectionError || error}
      onControl={control} onManualControl={manualControl} />} />;
}
