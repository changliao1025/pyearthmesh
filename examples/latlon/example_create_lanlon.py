"""Create a regular Lat-Lon mesh for the Qinghai Lake region.

This example uses the sectioned workflow configuration. The workflow compiler
translates it into the native Lat-Lon mesh settings before generation.
"""

from pathlib import Path

from pyearthmesh.config.config_manager import create_workflow


def main() -> None:
	config_file = Path(__file__).with_name("workflow_config.json")
	workflow = create_workflow(config_file)
	workflow.setup()
	mesh_cells = workflow.run()
	print(
		f"Created {len(mesh_cells)} Lat-Lon cells: "
		f"{workflow.case.config['sFilename_mesh']}"
	)


if __name__ == "__main__":
	main()
