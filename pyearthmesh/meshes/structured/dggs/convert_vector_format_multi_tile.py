import os, sys
from pyearth.toolbox.conversion.convert_vector_format import convert_vector_format
def convert_vector_format_multi_tile(sWorkspace_in, sWorkspace_out, sExtension_in, sExtension_out):
    #check existance of the input workspace
    if not os.path.exists(sWorkspace_in):
        print("The input workspace does not exist: " + sWorkspace_in)
        sys.exit(1)
    #create the output workspace if not exist
    if not os.path.exists(sWorkspace_out):
        os.makedirs(sWorkspace_out)

    #find all the matching files in the input workspace
    for root, dirs, files in os.walk(sWorkspace_in):
        for file in files:
            if file.endswith(sExtension_in):
                sFile_in = os.path.join(root, file)
                sFile_out = os.path.join(sWorkspace_out, os.path.splitext(file)[0] + sExtension_out)
                if os.path.exists(sFile_out):
                    #delete the existing output file before conversion
                    os.remove(sFile_out)
                print("Converting: " + sFile_in + " to " + sFile_out)
                #call the conversion function here
                convert_vector_format(sFile_in, sFile_out, use_ogr2ogr=True)

    print("Conversion completed for all matching files in the workspace.")
