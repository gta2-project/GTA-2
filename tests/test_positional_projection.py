"""
Unit tests for positional controller nullspace projection.

This test suite validates the sequential nullspace projection algorithm
used to combine multiple positional controllers while respecting priorities.

Algorithm:
- Controllers are processed in order (first = highest priority)
- Each controller's error is projected into the nullspace of all previous axes
- This ensures higher-priority controllers aren't disturbed by lower ones

Tests verify:
1. Single controller passes through unchanged
2. Two orthogonal controllers project correctly into nullspace
3. Zero-axis edge case is handled safely
4. Priority ordering (first controller has priority)
5. Mixed orthogonal and parallel components
6. Three orthogonal controllers filling all DoF

Run:
    python tests/test_positional_projection.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from gta.error_projection import project_errors, get_nullspace_matrix


def test_single_positional_controller():
    """Single positional controller should pass through unchanged."""
    packets = [
        {
            'type': 'positional',
            'axis': np.array([1.0, 0.0, 0.0]),  # X-axis
            'error': np.array([0.5, 0.0, 0.0])
        }
    ]
    
    pos_err, _, _, _ = project_errors(packets)
    
    # Should output the error unchanged
    np.testing.assert_allclose(pos_err, [0.5, 0.0, 0.0], atol=1e-6)
    print("✓ Single controller test passed")


def test_two_orthogonal_controllers():
    """Two orthogonal controllers should not interfere with each other."""
    packets = [
        {
            'type': 'positional',
            'axis': np.array([1.0, 0.0, 0.0]),  # X-axis (priority 1)
            'error': np.array([0.3, 0.0, 0.0])
        },
        {
            'type': 'positional',
            'axis': np.array([0.0, 1.0, 0.0]),  # Y-axis (priority 2)
            'error': np.array([0.0, 0.4, 0.0])
        }
    ]

    pos_err, _, _, _ = project_errors(packets)

    # Both errors should pass through (orthogonal)
    np.testing.assert_allclose(pos_err, [0.3, 0.4, 0.0], atol=1e-6)
    print("✓ Two orthogonal controllers test passed")


def test_parallel_controllers_priority():
    """When two controllers share the same axis, first has priority."""
    packets = [
        {
            'type': 'positional',
            'axis': np.array([1.0, 0.0, 0.0]),  # X-axis (priority 1)
            'error': np.array([0.5, 0.0, 0.0])
        },
        {
            'type': 'positional',
            'axis': np.array([1.0, 0.0, 0.0]),  # X-axis again (priority 2)
            'error': np.array([0.3, 0.0, 0.0])  # Should be projected out
        }
    ]
    
    pos_err, _, _, _ = project_errors(packets)
    
    # Only first controller's error should remain
    np.testing.assert_allclose(pos_err, [0.5, 0.0, 0.0], atol=1e-6)
    print("✓ Parallel controllers priority test passed")


def test_nullspace_projection_mixed():
    """Test nullspace projection with mixed orthogonal and parallel components."""
    packets = [
        {
            'type': 'positional',
            'axis': np.array([1.0, 0.0, 0.0]),  # X-axis (priority 1)
            'error': np.array([0.5, 0.0, 0.0])
        },
        {
            'type': 'positional',
            'axis': np.array([0.0, 1.0, 0.0]),  # Y-axis (priority 2)
            'error': np.array([0.2, 0.3, 0.0])  # Has X component that should be nulled
        }
    ]

    pos_err, _, _, _ = project_errors(packets)

    # First controller: full [0.5, 0, 0]
    # Second controller: [0.2, 0.3, 0] projected into nullspace of X
    # Expected: [0.5, 0, 0] + [0, 0.3, 0] = [0.5, 0.3, 0]
    np.testing.assert_allclose(pos_err, [0.5, 0.3, 0.0], atol=1e-6)
    print("✓ Mixed nullspace projection test passed")


def test_zero_axis_edge_case():
    """Controller with zero axis should be safely ignored."""
    packets = [
        {
            'type': 'positional',
            'axis': np.array([0.0, 0.0, 0.0]),  # Zero axis
            'error': np.array([0.5, 0.3, 0.2])
        }
    ]

    pos_err, _, _, _ = project_errors(packets)

    # Error should pass through but axis not added to nullspace basis
    np.testing.assert_allclose(pos_err, [0.5, 0.3, 0.2], atol=1e-6)
    print("✓ Zero axis edge case test passed")


def test_get_nullspace_matrix_empty():
    """Nullspace of empty axis list should be identity."""
    P = get_nullspace_matrix([])
    np.testing.assert_allclose(P, np.eye(3), atol=1e-6)
    print("✓ Empty nullspace matrix test passed")


def test_get_nullspace_matrix_single_axis():
    """Nullspace of single axis should project perpendicular components."""
    axes = [np.array([1.0, 0.0, 0.0])]  # X-axis
    P = get_nullspace_matrix(axes)
    
    # Should preserve Y and Z, zero out X
    test_vec = np.array([0.5, 0.3, 0.2])
    projected = P @ test_vec
    
    np.testing.assert_allclose(projected, [0.0, 0.3, 0.2], atol=1e-6)
    print("✓ Single axis nullspace matrix test passed")


def test_three_orthogonal_controllers():
    """Three orthogonal controllers filling all DoF."""
    packets = [
        {
            'type': 'positional',
            'axis': np.array([1.0, 0.0, 0.0]),  # X
            'error': np.array([0.1, 0.0, 0.0])
        },
        {
            'type': 'positional',
            'axis': np.array([0.0, 1.0, 0.0]),  # Y
            'error': np.array([0.0, 0.2, 0.0])
        },
        {
            'type': 'positional',
            'axis': np.array([0.0, 0.0, 1.0]),  # Z
            'error': np.array([0.0, 0.0, 0.3])
        }
    ]
    
    pos_err, _, _, _ = project_errors(packets)
    
    # All three errors should pass through
    np.testing.assert_allclose(pos_err, [0.1, 0.2, 0.3], atol=1e-6)
    print("✓ Three orthogonal controllers test passed")


def test_multi_axis_controller_claims_plane():
    """A planar controller should claim two independent positional DOFs."""
    packets = [
        {
            'type': 'positional',
            'axis': np.array([0.0, 0.0, 1.0]),
            'error': np.array([0.0, 0.0, 0.2]),
        },
        {
            'type': 'positional',
            'axes': [
                np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
            ],
            'error': np.array([0.3, 0.4, 0.5]),
        },
    ]

    pos_err, _, _, _ = project_errors(packets)
    np.testing.assert_allclose(pos_err, [0.3, 0.4, 0.2], atol=1e-6)
    print("✓ Multi-axis plane controller test passed")


def test_multi_axis_controller_projects_conflicting_components():
    """Higher-priority axes remove only conflicting planar components."""
    packets = [
        {
            'type': 'positional',
            'axis': np.array([1.0, 0.0, 0.0]),
            'error': np.array([0.1, 0.0, 0.0]),
        },
        {
            'type': 'positional',
            'axes': [
                np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
            ],
            'error': np.array([0.3, 0.4, 0.0]),
        },
        {
            'type': 'positional',
            'axis': np.array([0.0, 0.0, 1.0]),
            'error': np.array([0.0, 0.0, 0.5]),
        },
    ]

    pos_err, _, _, _ = project_errors(packets)
    np.testing.assert_allclose(pos_err, [0.1, 0.4, 0.5], atol=1e-6)
    print("✓ Multi-axis conflict projection test passed")


if __name__ == "__main__":
    print("Running positional projection tests...\n")
    
    test_single_positional_controller()
    test_two_orthogonal_controllers()
    test_parallel_controllers_priority()
    test_nullspace_projection_mixed()
    test_zero_axis_edge_case()
    test_get_nullspace_matrix_empty()
    test_get_nullspace_matrix_single_axis()
    test_three_orthogonal_controllers()
    test_multi_axis_controller_claims_plane()
    test_multi_axis_controller_projects_conflicting_components()
    
    print("\n✅ All positional projection tests passed!")
