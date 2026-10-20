export type Level = 'info' | 'warning' | 'error';

export interface EventItem {
  id: string;
  timestamp: string;
  type: string;
  sender: string;
  receiver: string;
  message_type: string;
  content: unknown;
  level: Level;
}

export interface ArtifactInfo {
  name: string;
  exists: boolean;
  url: string | null;
  size: number | null;
  mtime: string | null;
}

export interface Health {
  ready: boolean;
  initialized: boolean;
  init_error: string | null;
  camera_backend: string;
  gta_backend: 'ros1' | 'ros2';
  camera_image_path: string;
  agent_models: Record<
    string,
    {
      provider: string | null;
      model: string | null;
    }
  >;
  robot_dependencies: {
    rospy: boolean;
    rclpy: boolean;
    rtde_receive: boolean;
  };
}

export interface WorkflowState {
  task: string | null;
  subtasks: Record<string, unknown>[];
  skills: Record<string, unknown>[];
  parameters: Record<string, unknown>[];
  keypoints: Record<string, unknown>;
  messages: Record<string, unknown>[];
  events: EventItem[];
  artifacts: Record<string, ArtifactInfo>;
  busy_stage: string | null;
  last_error: string | null;
  script: string | null;
  script_path: string | null;
  revision: number;
  grounding_ready: boolean;
  script_ready: boolean;
  active_handlers: number;
  image_revision: number | null;
}

export interface Checkpoint {
  id: string;
  name: string;
  notes: string;
  task: string;
  visible: boolean;
  thumbnail_url: string | null;
  created_at: string;
}

export interface LibrarySession {
  id: string;
  checkpoint_id: string;
  checkpoint: Checkpoint;
}

export interface ExecutionState {
  operation?: 'run' | 'reset_joints' | 'open_gripper' | 'close_gripper' | null;
  status: 'idle' | 'starting_server' | 'preparing' | 'preview' | 'running' | 'stopping' | 'completed' | 'cancelled' | 'failed';
  run_id: string | null;
  session_id: string | null;
  subtask: string | null;
  subtasks: { id: string; description: string; status: string }[];
  skip_pending: boolean;
  error: string | null;
  logs: { role: string; text: string }[];
}
