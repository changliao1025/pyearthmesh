import os
from pyearthmesh.utility.mesh_utility import check_mesh_quality
def debug_idl(sFile_in, sFile_out, iFlag_drop_idl_crossing_cells):
    print("check mesh quality: " + sFile_in + " to " + sFile_out)
    sFilename_mesh_new = check_mesh_quality(sFile_in, iFlag_verbose_in=1, iFlag_drop_idl_crossing_cells_in=iFlag_drop_idl_crossing_cells)
    print("check mesh quality completed: " + sFilename_mesh_new)

    if sFilename_mesh_new == sFile_in:
        # make a copy of the original file to the output location
        shutil.copy2(sFile_in, sFile_out)
        print(f"Original mesh file copied to: {sFile_out}")
    else:
        # replace target atomically when possible
        os.replace(sFilename_mesh_new, sFile_out)
        print(f"New mesh file moved to: {sFile_out}")

if __name__ == "__main__":
    sFile_in=r'C:\Users\chang\scratch\04model\pyearthmesh\healpix\50km\Parquet\01\healpix_mesh_base_01.parquet'
    sFile_out=r'C:\Users\chang\scratch\04model\pyearthmesh\healpix\50km\Parquet\01\healpix_mesh_base_01_debug.parquet'
    iFlag_drop_idl_crossing_cells=0

    debug_idl(sFile_in, sFile_out, iFlag_drop_idl_crossing_cells)
