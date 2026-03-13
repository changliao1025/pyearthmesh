import shutil
import os, sys
from concurrent.futures import ProcessPoolExecutor
from pyearth.toolbox.conversion.convert_vector_format import convert_vector_format
from pyearthmesh.utility.mesh_utility import check_mesh_quality


def _check_single_mesh_quality(task):
    """Run quality check for one mesh file and write output file."""
    sFile_in, sWorkspace_out, sExtension_out , iFlag_drop_idl_crossing_cells = task
    sFile_out = os.path.join(
        sWorkspace_out,
        os.path.splitext(os.path.basename(sFile_in))[0] + sExtension_out,
    )

    if os.path.exists(sFile_out):
        # delete the existing output file before conversion
        os.remove(sFile_out)

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

    return sFile_out


def check_mesh_quality_multi_tile(
    sWorkspace_in,
    sWorkspace_out,
    sExtension_in,
    sExtension_out,
    iParallel_in=0,
    nWorker_in=None,
):
    #check existance of the input workspace
    if not os.path.exists(sWorkspace_in):
        print("The input workspace does not exist: " + sWorkspace_in)
        sys.exit(1)
    #create the output workspace if not exist
    if not os.path.exists(sWorkspace_out):
        os.makedirs(sWorkspace_out)

    #find all the matching files in the input workspace
    sWorkspace_in_abs = os.path.abspath(sWorkspace_in)
    sWorkspace_out_abs = os.path.abspath(sWorkspace_out)

    aTask = []
    for root, dirs, files in os.walk(sWorkspace_in):
        # process only the top-level input folder (no recursion)
        if os.path.abspath(root) != sWorkspace_in_abs:
            continue

        # disable descent into subfolders
        dirs[:] = []

        # skip processing if current root is the output workspace (safety)
        if os.path.abspath(root).startswith(sWorkspace_out_abs):
            continue
        #sort files to ensure consistent processing order
        files.sort()
        #exclude files that has '_fixed' in the name to avoid re-processing already fixed meshes
        files = [f for f in files if '_fixed' not in f]
        for file in files:
            if file.endswith(sExtension_in):
                sFile_in = os.path.join(root, file)
                if '06' in sFile_in:
                    #all ocean cell, so we dont need to worry about IDL crossing cells
                    iFlag_drop_idl_crossing_cells_in = True
                else:
                    iFlag_drop_idl_crossing_cells_in = False
                aTask.append((sFile_in, sWorkspace_out, sExtension_out, iFlag_drop_idl_crossing_cells_in))

        # top-level already processed; stop walking
        break

    nFile = len(aTask)
    print(f"Found {nFile} matching files in the input workspace: {sWorkspace_in}")

    if iParallel_in and nFile > 1:
        with ProcessPoolExecutor(max_workers=nWorker_in) as executor:
            for _ in executor.map(_check_single_mesh_quality, aTask):
                pass
    else:
        for task in aTask:
            _check_single_mesh_quality(task)

    print("Check mesh quality completed for all matching files in the workspace.")
