"""Lossless JSON encoding for agent checkpoints (never serialize SDK clients)."""

import base64
import copy
from datetime import datetime
import numpy as np
from communication import Message


STATE_FIELDS = (
    'state', 'current_task', 'current_subtasks', 'current_skills',
    'feedback_history', 'keypoint_locations', 'current_keypoints',
    'current_queries', 'keypoint_descriptions', 'last_image_np', 'last_image_size',
)


def encode(value):
    if isinstance(value, bytes):
        return {'__gta_type__': 'bytes', 'data': base64.b64encode(value).decode('ascii')}
    if isinstance(value, np.ndarray):
        return {'__gta_type__': 'array', 'dtype': str(value.dtype),
                'shape': list(value.shape), 'data': encode(value.tobytes())}
    if isinstance(value, Message):
        return {'__gta_type__': 'message', 'data': value.to_dict()}
    if isinstance(value, tuple):
        return {'__gta_type__': 'tuple', 'data': [encode(v) for v in value]}
    if hasattr(value, 'model_dump'):
        return encode(value.model_dump(mode='python', exclude_none=True))
    if isinstance(value, dict):
        return {k: encode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [encode(v) for v in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f'Cannot checkpoint {type(value).__name__}')


def decode(value):
    if isinstance(value, list):
        return [decode(v) for v in value]
    if not isinstance(value, dict):
        return value
    kind = value.get('__gta_type__')
    if kind == 'bytes':
        return base64.b64decode(value['data'], validate=True)
    if kind == 'array':
        return np.frombuffer(decode(value['data']), dtype=value['dtype']).reshape(value['shape']).copy()
    if kind == 'tuple':
        return tuple(decode(v) for v in value['data'])
    if kind == 'message':
        return restore_message(value['data'])
    return {k: decode(v) for k, v in value.items()}


def restore_message(data):
    message = Message(data['sender'], data['receiver'], data['message_type'], data['content'])
    message.id = data['id']
    message.timestamp = datetime.fromisoformat(data['timestamp'])
    return message


def openai_input(item):
    """Convert exported native items to independent conversation inputs."""
    item = copy.deepcopy(item)
    # Native assistant outputs and reasoning inputs require their IDs, status,
    # content fields and phase. They are content inserts, not item references.
    if item.get('type') == 'message' and item['role'] != 'assistant':
        item.pop('id', None)
        item.pop('status', None)
    return item
