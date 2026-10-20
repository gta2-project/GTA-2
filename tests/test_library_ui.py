"""Optional browser layout checks against built assets, with every API mocked.

Build frontend first. Install Playwright and Chromium for these checks, or set
GTA_TEST_BROWSER_EXECUTABLE to an existing compatible Chromium binary.
"""
import json
import mimetypes
import os
from pathlib import Path
from urllib.parse import urlparse

import pytest

playwright = pytest.importorskip('playwright.sync_api')
DIST = Path(__file__).resolve().parents[1] / 'frontend' / 'dist'


@pytest.fixture
def browser_page():
    if not (DIST / 'library.html').is_file():
        pytest.skip('Build both frontend entry points before browser tests')
    with playwright.sync_playwright() as runtime:
        options = {'headless': True}
        if os.getenv('GTA_TEST_BROWSER_EXECUTABLE'):
            options['executable_path'] = os.environ['GTA_TEST_BROWSER_EXECUTABLE']
        try:
            browser = runtime.chromium.launch(**options)
        except playwright.Error as exc:
            pytest.skip(f'Chromium is not installed: {exc}')
        page = browser.new_page(viewport={'width': 1440, 'height': 1000})
        # Do not create a real websocket or contact any camera, model or robot.
        page.add_init_script('window.WebSocket = class { close() {} };')
        errors = []
        requests = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        checkpoint = {'id': 'a' * 32, 'name': 'Prepared box', 'notes': 'Place the box on the table',
                      'task': 'Move above the box', 'visible': True,
                      'thumbnail_url': 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7',
                      'created_at': '2026-09-22T12:00:00Z'}
        session = {'id': 'b' * 32, 'checkpoint_id': checkpoint['id'], 'checkpoint': checkpoint}
        state = {'task': checkpoint['task'], 'subtasks': [], 'skills': [], 'parameters': [],
                 'keypoints': {}, 'messages': [], 'events': [], 'artifacts': {},
                 'busy_stage': None, 'last_error': None, 'script': 'print("reviewed")',
                 'script_path': 'robot_script.py', 'revision': 1, 'image_revision': 1,
                 'grounding_ready': True, 'script_ready': True, 'active_handlers': 0}
        health = {'ready': True, 'initialized': True, 'init_error': None, 'camera_backend': 'ros2', 'gta_backend': 'ros2',
                  'agent_models': {}, 'robot_dependencies': {'rclpy': True, 'rospy': False, 'rtde_receive': True}}
        execution = {'status': 'idle', 'run_id': None, 'session_id': None, 'subtask': None,
                     'subtasks': [], 'logs': [], 'error': None, 'skip_pending': False}

        def route(request):
            path = urlparse(request.request.url).path
            requests.append((request.request.method, path))
            if path.startswith('/api/'):
                if path == '/api/checkpoints':
                    value = [checkpoint]
                elif path == '/api/library/session':
                    value = session
                elif path == '/api/execution':
                    value = execution
                elif path.endswith('/state'):
                    value = state
                elif path.endswith('/health'):
                    value = health
                else:
                    request.fulfill(status=404, body='{}', content_type='application/json')
                    return
                request.fulfill(body=json.dumps(value), content_type='application/json')
                return
            file = DIST / ('index.html' if path == '/' else 'library.html')
            if path.startswith('/assets/'):
                file = DIST / path.lstrip('/')
            request.fulfill(body=file.read_bytes(), content_type=mimetypes.guess_type(str(file))[0] or 'application/octet-stream')

        page.route('**/*', route)
        yield page, requests
        assert not errors
        assert not any(method == 'POST' for method, _ in requests)
        browser.close()


@pytest.mark.parametrize('camera_backend,gta_backend,label', [
    ('file', 'ros2', 'ROS 2 available'),
    ('ros2', 'ros1', 'ROS available'),
])
def test_middleware_status_uses_gta_backend_not_camera(browser_page, camera_backend, gta_backend, label):
    page, _ = browser_page
    health = {'ready': True, 'initialized': True, 'init_error': None,
              'camera_backend': camera_backend, 'gta_backend': gta_backend,
              'agent_models': {}, 'robot_dependencies': {
                  'rclpy': gta_backend == 'ros2', 'rospy': gta_backend == 'ros1', 'rtde_receive': True}}
    page.route('**/api/health', lambda route: route.fulfill(body=json.dumps(health), content_type='application/json'))
    page.goto('http://gta.test/')
    playwright.expect(page.get_by_text(label, exact=True)).to_be_visible()


def script_panel(page):
    page.locator('.stage-tab').filter(has_text='Robot Script').click()
    return page.locator('.domain-panel').filter(has=page.get_by_role('heading', name='Robot Script', exact=True))


def test_normal_dashboard_has_robot_controls_without_library_navigation(browser_page):
    page, requests = browser_page
    page.goto('http://gta.test/')
    playwright.expect(page.get_by_role('heading', name='GTA Sessions', exact=True)).to_be_visible()
    panel = script_panel(page)
    playwright.expect(panel.get_by_role('button', name='Save Workflow', exact=True)).to_be_visible()
    footer = panel.get_by_role('region', name='Robot execution')
    playwright.expect(footer).to_be_visible()
    for label in ('Run', 'Skip Subtask', 'Cancel', 'Reset Joints', 'Open Gripper', 'Close Gripper'):
        playwright.expect(footer.get_by_role('button', name=label, exact=True)).to_be_visible()
    preview = panel.locator('.script-preview')
    assert footer.bounding_box()['y'] >= preview.bounding_box()['y'] + preview.bounding_box()['height']
    assert page.locator('.library-tabs, .checkpoint-save, .refinement-panel').count() == 0
    assert page.get_by_role('button', name='Reset to Checkpoint').count() == 0
    assert not any(path in {'/api/checkpoints', '/api/library/session'} for _, path in requests)


@pytest.mark.parametrize('path,control', [('/admin', 'Save Checkpoint'), ('/run', 'Robot execution')])
def test_library_controls_are_below_script_inside_its_panel(browser_page, path, control):
    page, _ = browser_page
    page.goto('http://gta.test' + path)
    playwright.expect(page.locator('.stage-browser-tabs')).to_be_visible()
    assert page.locator('.library-tabs a').all_text_contents() == ['Prepare', 'Library', 'Run']
    assert page.locator('header, .library-nav, .library-session-header').count() == 0
    panel = script_panel(page)
    footer = panel.get_by_role('region', name=control, exact=True)
    playwright.expect(footer).to_be_visible()
    preview = panel.locator('.script-preview')
    assert footer.bounding_box()['y'] >= preview.bounding_box()['y'] + preview.bounding_box()['height']
    if path == '/run':
        playwright.expect(footer.get_by_role('button', name='Run', exact=True)).to_be_enabled()
        for label in ('Skip Subtask', 'Cancel', 'Reset Joints', 'Open Gripper', 'Close Gripper', 'Reset to Checkpoint'):
            playwright.expect(footer.get_by_role('button', name=label, exact=True)).to_be_visible()


@pytest.mark.parametrize('target,button,path', [
    ('run', 'Load in Run', '/run'),
    ('admin', 'Open in Prepare', '/admin'),
])
def test_library_load_uses_session_api_and_navigates_without_motion(browser_page, target, button, path):
    page, requests = browser_page
    calls = []

    def open_session(route):
        calls.append(route.request.post_data_json)
        route.fulfill(body=json.dumps({'id': 'b' * 32, 'checkpoint_id': 'a' * 32}),
                      content_type='application/json')

    page.route('**/api/library/sessions', open_session)
    page.goto('http://gta.test/library')
    page.get_by_role('button', name=button, exact=True).click()
    playwright.expect(page).to_have_url('http://gta.test' + path)
    assert calls == [{'checkpoint_id': 'a' * 32, 'target': target}]
    assert not any(method == 'POST' for method, _ in requests)


@pytest.mark.parametrize('label,operation', [('Reset Joints', 'reset_joints'), ('Open Gripper', 'open_gripper'), ('Close Gripper', 'close_gripper')])
@pytest.mark.parametrize('path', ['/', '/run'])
def test_manual_robot_buttons_post_correct_operation_and_keep_cancel_available(browser_page, label, operation, path):
    page, _ = browser_page
    calls = []
    execution = {'status': 'running', 'operation': operation, 'run_id': 'manual-test',
                 'session_id': 'main' if path == '/' else 'b' * 32, 'subtask': None, 'subtasks': [],
                 'logs': [], 'error': None, 'skip_pending': False}
    def manual_route(route):
        calls.append(route.request.post_data_json)
        page.route('**/api/execution', lambda request: request.fulfill(body=json.dumps(execution), content_type='application/json'))
        route.fulfill(body=json.dumps(execution), content_type='application/json')
    page.route('**/execution/manual', manual_route)
    page.on('dialog', lambda dialog: dialog.accept())
    page.goto('http://gta.test' + path)
    panel = script_panel(page)
    panel.get_by_role('button', name=label, exact=True).click()
    playwright.expect(panel.get_by_text(f'{label} · running', exact=True)).to_be_visible()
    assert calls == [{'operation': operation, 'confirmed': operation == 'reset_joints'}]
    for button in ('Run', 'Skip Subtask', 'Reset Joints', 'Open Gripper', 'Close Gripper'):
        playwright.expect(panel.get_by_role('button', name=button, exact=True)).to_be_disabled()
    if path == '/run':
        playwright.expect(panel.get_by_role('button', name='Reset to Checkpoint')).to_be_disabled()
    playwright.expect(panel.get_by_role('button', name='Cancel', exact=True)).to_be_enabled()


@pytest.mark.parametrize('path', ['/', '/run'])
def test_dismissing_joint_reset_confirmation_sends_no_command(browser_page, path):
    page, requests = browser_page
    page.on('dialog', lambda dialog: dialog.dismiss())
    page.goto('http://gta.test' + path)
    script_panel(page).get_by_role('button', name='Reset Joints', exact=True).click()
    assert not any(method == 'POST' for method, _ in requests)


@pytest.mark.parametrize('path', ['/', '/run'])
def test_manual_controls_are_enabled_without_generated_script(browser_page, path):
    page, _ = browser_page
    def no_script(route):
        state = {'task': '', 'subtasks': [], 'skills': [], 'parameters': [], 'keypoints': {},
                 'messages': [], 'events': [], 'artifacts': {}, 'busy_stage': None,
                 'last_error': None, 'script': None, 'script_path': None,
                 'revision': 1, 'grounding_ready': False, 'script_ready': False, 'active_handlers': 0}
        route.fulfill(body=json.dumps(state), content_type='application/json')
    page.route('**/state', no_script)
    page.goto('http://gta.test' + path)
    panel = script_panel(page)
    playwright.expect(panel.get_by_role('button', name='Run', exact=True)).to_be_disabled()
    for label in ('Reset Joints', 'Open Gripper', 'Close Gripper'):
        playwright.expect(panel.get_by_role('button', name=label, exact=True)).to_be_enabled()


def test_main_run_preview_skip_cancel_and_logs(browser_page):
    page, _ = browser_page
    execution = {'status': 'idle', 'operation': None, 'run_id': None, 'session_id': None,
                 'subtask': None, 'subtasks': [], 'logs': [], 'error': None, 'skip_pending': False}
    calls = []
    page.route('**/api/execution', lambda route: route.fulfill(body=json.dumps(execution), content_type='application/json'))
    def control(route):
        operation = urlparse(route.request.url).path.rsplit('/', 1)[1]
        calls.append((operation, route.request.post_data_json))
        if operation == 'run':
            execution.update(status='starting_server', operation='run', run_id='main-run', session_id='main')
        elif operation == 'skip':
            execution['skip_pending'] = True
        elif operation == 'cancel':
            execution['status'] = 'stopping'
        route.fulfill(body=json.dumps(execution), content_type='application/json')
    page.route('**/api/execution/*', control)
    page.goto('http://gta.test/')
    panel = script_panel(page)
    playwright.expect(panel.get_by_role('button', name='Run', exact=True)).to_be_enabled()
    panel.get_by_role('button', name='Run', exact=True).click()
    playwright.expect(panel.get_by_text('Waiting for robot and gripper readiness…')).to_be_visible()
    playwright.expect(page.locator('#task')).to_be_disabled()
    playwright.expect(panel.get_by_role('button', name='Generate Script', exact=True)).to_be_disabled()
    execution['status'] = 'preview'
    playwright.expect(panel.get_by_text('Review 3D preview—close the window to continue')).to_be_visible()
    playwright.expect(panel.get_by_role('button', name='Skip Subtask')).to_be_disabled()
    playwright.expect(panel.get_by_role('button', name='Cancel')).to_be_enabled()
    execution.update(status='running', subtask='1', subtasks=[{'id': '1', 'description': 'Reviewed move', 'status': 'running'}],
                     logs=[{'role': 'server', 'text': 'Ready for skill'}])
    playwright.expect(panel.get_by_role('button', name='Skip Subtask')).to_be_enabled()
    panel.get_by_role('button', name='Skip Subtask').click()
    playwright.expect(panel.get_by_role('button', name='Skip Subtask')).to_be_disabled()
    panel.get_by_role('button', name='Cancel').click()
    playwright.expect(panel.get_by_text('Stopping owned processes and verifying hardware cleanup…')).to_be_visible()
    playwright.expect(panel.get_by_role('button', name='Run', exact=True)).to_be_disabled()
    playwright.expect(panel.get_by_role('button', name='Cancel')).to_be_disabled()
    panel.locator('summary').filter(has_text='Robot execution logs').click()
    playwright.expect(panel.locator('.execution-logs')).to_have_text('[server] Ready for skill')
    assert calls == [('run', {'run_id': None, 'subtask': None}),
                     ('skip', {'run_id': 'main-run', 'subtask': '1'}),
                     ('cancel', {'run_id': 'main-run', 'subtask': '1'})]
    execution.update(status='cancelled', subtask=None, skip_pending=False)
    playwright.expect(panel.get_by_role('button', name='Run', exact=True)).to_be_enabled()


@pytest.mark.parametrize('status', ['preview', 'running', 'stopping'])
def test_main_refresh_reconnects_without_starting_motion(browser_page, status):
    page, _ = browser_page
    execution = {'status': status, 'operation': 'run', 'run_id': 'existing', 'session_id': 'main',
                 'subtask': '1' if status == 'running' else None, 'subtasks': [],
                 'logs': [], 'error': None, 'skip_pending': False}
    page.route('**/api/execution', lambda route: route.fulfill(body=json.dumps(execution), content_type='application/json'))
    page.goto('http://gta.test/')
    for _ in range(2):
        panel = script_panel(page)
        playwright.expect(panel.get_by_text(f'Robot · {status}', exact=True)).to_be_visible()
        playwright.expect(panel.get_by_role('button', name='Run', exact=True)).to_be_disabled()
        playwright.expect(page.locator('#task')).to_be_disabled()
        if status == 'stopping':
            playwright.expect(panel.get_by_role('button', name='Cancel')).to_be_disabled()
        else:
            playwright.expect(panel.get_by_role('button', name='Cancel')).to_be_enabled()
        page.reload()


def test_main_robot_api_errors_are_visible_inside_script_panel(browser_page):
    page, _ = browser_page
    page.route('**/api/execution/run', lambda route: route.fulfill(status=409,
               body=json.dumps({'detail': 'Another GTA dashboard is controlling the robot'}), content_type='application/json'))
    page.goto('http://gta.test/')
    panel = script_panel(page)
    panel.get_by_role('button', name='Run', exact=True).click()
    playwright.expect(panel.get_by_role('alert')).to_have_text('Another GTA dashboard is controlling the robot')


def test_main_controls_wait_for_background_handlers(browser_page):
    page, _ = browser_page
    state = {'task': '', 'subtasks': [], 'skills': [], 'parameters': [], 'keypoints': {},
             'messages': [], 'events': [], 'artifacts': {}, 'busy_stage': None,
             'last_error': None, 'script': 'print("reviewed")', 'script_path': 'robot_script.py',
             'revision': 1, 'grounding_ready': True, 'script_ready': True, 'active_handlers': 1}
    page.route('**/api/state', lambda route: route.fulfill(body=json.dumps(state), content_type='application/json'))
    page.goto('http://gta.test/')
    panel = script_panel(page)
    for label in ('Run', 'Generate Script', 'Reset Joints', 'Open Gripper', 'Close Gripper'):
        playwright.expect(panel.get_by_role('button', name=label, exact=True)).to_be_disabled()


def test_loaded_library_has_no_preparation_reference_panel(browser_page):
    page, requests = browser_page
    page.goto('http://gta.test/run')
    playwright.expect(page.locator('.stage-browser-tabs')).to_be_visible()
    playwright.expect(page.locator('#library-checkpoint')).to_have_value('a' * 32)
    assert page.locator('.preparation-reference').count() == 0
    assert page.get_by_role('heading', name='Prepared box', exact=True).count() == 0
    assert page.get_by_role('img', name='Preparation reference', exact=True).count() == 0
    assert page.get_by_text('Preparation reference — not the current scene', exact=True).count() == 0
    assert not any('/reference/' in path for _, path in requests)


def test_library_keeps_requery_and_generation_only_in_panels(browser_page):
    page, _ = browser_page
    page.goto('http://gta.test/run')
    playwright.expect(page.locator('.stage-browser-tabs')).to_be_visible()
    assert page.locator('.library-readiness').count() == 0
    assert page.get_by_role('button', name='Requery Scene', exact=True).count() == 0

    page.locator('.stage-tab').filter(has_text='Vision Module').click()
    vision = page.locator('.domain-panel').filter(
        has=page.get_by_role('heading', name='Vision & Keypoints', exact=True))
    playwright.expect(vision.get_by_role('button', name='Requery', exact=True)).to_be_visible()
    playwright.expect(vision.get_by_role('button', name='Requery', exact=True)).to_be_enabled()

    panel = script_panel(page)
    playwright.expect(panel.get_by_role('button', name='Generate Script', exact=True)).to_be_visible()
    playwright.expect(panel.get_by_role('button', name='Generate Script', exact=True)).to_be_enabled()
    assert page.get_by_role('button', name='Generate Script', exact=True).count() == 1


@pytest.mark.parametrize('has_session', [True, False])
def test_library_checkpoint_selector_is_top_right_beside_navigation(browser_page, has_session):
    page, _ = browser_page
    if not has_session:
        page.route('**/api/library/session', lambda route: route.fulfill(
            body='null', content_type='application/json'))
    page.goto('http://gta.test/run')
    toolbar = page.locator('.library-toolbar')
    selection = toolbar.get_by_role('region', name='Load checkpoint')
    playwright.expect(selection.get_by_label('Prepared checkpoint')).to_be_visible()
    playwright.expect(selection.get_by_role('button', name='Load', exact=True)).to_be_visible()
    nav_box = toolbar.get_by_role('navigation', name='Library pages').bounding_box()
    selection_box = selection.bounding_box()
    toolbar_box = toolbar.bounding_box()
    assert selection_box['x'] > nav_box['x'] + nav_box['width']
    assert abs((selection_box['y'] + selection_box['height'] / 2)
               - (nav_box['y'] + nav_box['height'] / 2)) < 2
    assert abs((selection_box['x'] + selection_box['width'])
               - (toolbar_box['x'] + toolbar_box['width'])) < 2

    page.set_viewport_size({'width': 375, 'height': 812})
    playwright.expect(selection.get_by_label('Prepared checkpoint')).to_be_visible()
    playwright.expect(selection.get_by_role('button', name='Load', exact=True)).to_be_visible()
    assert toolbar.evaluate('(element) => element.scrollWidth <= element.clientWidth')


def test_library_is_its_own_page(browser_page):
    page, _ = browser_page
    page.goto('http://gta.test/library')
    playwright.expect(page.get_by_role('heading', name='Library', exact=True)).to_be_visible()
    playwright.expect(page.get_by_role('heading', name='Prepared box', exact=True)).to_be_visible()
    playwright.expect(page.get_by_role('img', name='Preparation: Prepared box', exact=True)).to_be_visible()
    playwright.expect(page.get_by_role('button', name='Open in Prepare')).to_be_visible()
    playwright.expect(page.get_by_role('button', name='Load in Run')).to_be_visible()
    playwright.expect(page.get_by_role('switch', name='Show in Run')).to_be_checked()
    assert page.locator('.workspace, .checkpoint-save, .execution-panel').count() == 0


def test_library_search_and_filters(browser_page):
    page, _ = browser_page
    page.goto('http://gta.test/library')
    search = page.get_by_role('searchbox', name='Search saved plans')
    search.fill('  TABLE  ')
    playwright.expect(page.get_by_role('article', name='Prepared box')).to_be_visible()
    search.fill('missing task')
    playwright.expect(page.get_by_role('heading', name='No matching plans')).to_be_visible()
    assert page.locator('.checkpoint-card').count() == 0
    page.get_by_role('button', name='Clear search', exact=True).click()
    playwright.expect(page.get_by_role('article', name='Prepared box')).to_be_visible()
    page.get_by_label('Filter plans by visibility').select_option('hidden')
    playwright.expect(page.get_by_role('heading', name='No matching plans')).to_be_visible()
    page.get_by_role('button', name='Clear filters', exact=True).click()
    playwright.expect(page.get_by_role('article', name='Prepared box')).to_be_visible()


def test_library_hidden_plan_and_missing_image(browser_page):
    page, _ = browser_page
    checkpoint = {'id': 'c' * 32, 'name': 'Hidden plan', 'task': 'Review before using', 'notes': '',
                  'visible': False, 'thumbnail_url': None, 'created_at': '2026-09-22T12:00:00Z'}
    page.route('**/api/checkpoints?*', lambda route: route.fulfill(
        body=json.dumps([checkpoint]), content_type='application/json'))
    page.goto('http://gta.test/library')
    playwright.expect(page.get_by_text('No preparation image', exact=True)).to_be_visible()
    playwright.expect(page.get_by_role('button', name='Load in Run')).to_be_disabled()
    playwright.expect(page.get_by_role('button', name='Open in Prepare')).to_be_enabled()
    playwright.expect(page.get_by_role('switch', name='Show in Run')).not_to_be_checked()
    page.get_by_label('Filter plans by visibility').select_option('hidden')
    playwright.expect(page.get_by_role('article', name='Hidden plan')).to_be_visible()


def test_library_empty_state(browser_page):
    page, _ = browser_page
    page.route('**/api/checkpoints?*', lambda route: route.fulfill(body='[]', content_type='application/json'))
    page.goto('http://gta.test/library')
    playwright.expect(page.get_by_role('heading', name='Your first plan starts in Prepare')).to_be_visible()
    playwright.expect(page.get_by_role('link', name='Prepare a plan')).to_have_attribute('href', '/admin')
    assert page.locator('.checkpoint-card').count() == 0


@pytest.mark.parametrize('width', [1440, 768, 375, 320])
def test_library_card_layout_is_responsive(browser_page, width):
    page, _ = browser_page
    page.set_viewport_size({'width': width, 'height': 1000})
    page.goto('http://gta.test/library')
    card = page.get_by_role('article', name='Prepared box')
    playwright.expect(card).to_be_visible()
    image_box = card.locator('.checkpoint-image').bounding_box()
    content_box = card.locator('.checkpoint-content').bounding_box()
    if width > 760:
        assert content_box['x'] >= image_box['x'] + image_box['width']
        assert content_box['y'] == image_box['y']
    else:
        assert content_box['y'] >= image_box['y'] + image_box['height']
    for button in card.get_by_role('button').all():
        playwright.expect(button).to_be_visible()
        bounds = button.bounding_box()
        assert 0 <= bounds['x'] < bounds['x'] + bounds['width'] <= width
    assert card.evaluate('(element) => element.scrollWidth <= element.clientWidth')
    assert page.locator('main').evaluate('(element) => element.scrollWidth <= element.clientWidth')


def test_library_actions_are_disabled_during_execution(browser_page):
    page, _ = browser_page
    page.route('**/api/execution', lambda route: route.fulfill(
        body=json.dumps({'status': 'running'}), content_type='application/json'))
    page.goto('http://gta.test/library')
    card = page.get_by_role('article', name='Prepared box')
    for label in ('Load in Run', 'Open in Prepare', 'Rename'):
        playwright.expect(card.get_by_role('button', name=label, exact=True)).to_be_disabled()
    playwright.expect(card.get_by_role('switch', name='Show in Run')).to_be_disabled()


def test_library_visibility_and_rename_use_existing_update_api(browser_page):
    page, _ = browser_page
    checkpoint = {'id': 'd' * 32, 'name': 'Saved plan', 'task': 'Review before using', 'notes': '',
                  'visible': True, 'thumbnail_url': None, 'created_at': '2026-09-22T12:00:00Z'}
    changes = []
    page.route('**/api/checkpoints?*', lambda route: route.fulfill(
        body=json.dumps([checkpoint]), content_type='application/json'))

    def update(route):
        assert route.request.method == 'PATCH'
        changes.append(route.request.post_data_json)
        checkpoint.update(route.request.post_data_json)
        route.fulfill(body=json.dumps(checkpoint), content_type='application/json')

    page.route('**/api/checkpoints/' + checkpoint['id'], update)
    page.goto('http://gta.test/library')
    toggle = page.get_by_role('switch', name='Show in Run')
    toggle.click()
    playwright.expect(toggle).not_to_be_checked()
    playwright.expect(page.get_by_role('button', name='Load in Run')).to_be_disabled()
    toggle.click()
    playwright.expect(toggle).to_be_checked()
    playwright.expect(page.get_by_role('button', name='Load in Run')).to_be_enabled()
    page.once('dialog', lambda dialog: dialog.accept('Renamed plan'))
    page.get_by_role('button', name='Rename', exact=True).click()
    playwright.expect(page.get_by_role('article', name='Renamed plan')).to_be_visible()
    assert changes == [{'visible': False}, {'visible': True}, {'name': 'Renamed plan'}]
