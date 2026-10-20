"""Structural validation shared by web checkpoints and generated scripts."""
import json
import re


def validate_plan(plan):
    if not isinstance(plan, list) or not plan or not all(isinstance(skill, dict) for skill in plan):
        raise ValueError('A completed parameterized plan is required')
    ids = set()
    for skill in plan:
        subtask_id = str(skill.get('subtask_id', ''))
        if not re.fullmatch(r'[1-9][0-9]*', subtask_id) or subtask_id in ids:
            raise ValueError('Subtasks require unique positive integer IDs')
        ids.add(subtask_id)
        if not isinstance(skill.get('controllers'), list) or not skill['controllers']:
            raise ValueError('Every subtask must have controllers')
        for controller in skill['controllers']:
            if not isinstance(controller, dict) or not isinstance(controller.get('controller'), str):
                raise ValueError('Each controller needs a controller expression')
            for parameter in controller.get('offset_parameters', []):
                if parameter.get('value') is None:
                    raise ValueError('All offset parameters must have assigned values')
    json.dumps(plan, allow_nan=False)
