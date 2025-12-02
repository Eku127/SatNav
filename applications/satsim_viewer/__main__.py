"""Entry point for running interactive viewer as a module.

Usage:
    # Run free viewer (default)
    python -m applications.satsim_viewer
    python -m applications.satsim_viewer free
    
    # Run task viewer
    python -m applications.satsim_viewer task
    
    # Run task viewer with custom config
    python -m applications.satsim_viewer task --config /path/to/vln_task.yaml
"""

import sys


def main():
    """Main entry point with viewer selection."""
    # Check if first argument is 'task' or 'free'
    if len(sys.argv) > 1 and sys.argv[1] in ["task", "free"]:
        viewer_choice = sys.argv[1]
        # Remove viewer choice from argv so it doesn't interfere with viewer's own argparse
        sys.argv = [sys.argv[0]] + sys.argv[2:]
    else:
        viewer_choice = "free"
    
    if viewer_choice == "free":
        from .free_viewer import main as free_main
        free_main()
    elif viewer_choice == "task":
        from .task_viewer import main as task_main
        task_main()


if __name__ == "__main__":
    main()

