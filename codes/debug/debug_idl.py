import os
import shutil

from pyearthmesh.utility.mesh_utility import check_mesh_quality


def _remove_existing_output_if_needed(path: str) -> None:
    """Delete an existing output file if present.

    On Windows, overwrite attempts can fail if another process still has the file open.
    We surface a clear PermissionError instead of failing with a vague OS message.
    """
    if os.path.exists(path):
        try:
            os.remove(path)
        except PermissionError as exc:
            raise PermissionError(
                f"Cannot overwrite output file because it is locked by another process: {path}"
            ) from exc


def debug_idl(sFile_in, sFile_out, iFlag_drop_idl_crossing_cells):
    print("check mesh quality: " + sFile_in + " to " + sFile_out)
    sFilename_mesh_new = check_mesh_quality(
        sFile_in,
        iFlag_verbose_in=1,
        iFlag_drop_idl_crossing_cells_in=iFlag_drop_idl_crossing_cells,
    )
    print("check mesh quality completed: " + sFilename_mesh_new)

    if sFilename_mesh_new == sFile_in:
        # make a copy of the original file to the output location
        _remove_existing_output_if_needed(sFile_out)
        shutil.copy2(sFile_in, sFile_out)
        print(f"Original mesh file copied to: {sFile_out}")
    else:
        # replace target atomically when possible
        _remove_existing_output_if_needed(sFile_out)
        os.replace(sFilename_mesh_new, sFile_out)
        print(f"New mesh file moved to: {sFile_out}")

if __name__ == "__main__":
    sFile_in=r'D:\scratch\04model\pyearthmesh\healpix\50km\Parquet\01\healpix_mesh_base_01.parquet'
    sFile_out=r'D:\scratch\04model\pyearthmesh\healpix\50km\Parquet\01\healpix_mesh_base_01_debug.parquet'
    iFlag_drop_idl_crossing_cells=0

    debug_idl(sFile_in, sFile_out, iFlag_drop_idl_crossing_cells)
