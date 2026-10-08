
"""Create a Healpix mesh using the unified JSON workflow configuration."""

from pathlib import Path

from pyearthmesh.config.config_manager import create_workflow


def main() -> None:
    config_file = Path(__file__).with_name("workflow_config.json")
    workflow = create_workflow(config_file)
    workflow.setup()
    result = workflow.run()
    print(f"Created Healpix mesh in {workflow.case.config['sWorkspace_output']}")
    print(f"Generator result: {result}")


if __name__ == "__main__":
    main()
