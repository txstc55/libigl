import sys
import os
import fileinput

if __name__ == '__main__':
    # Modify Cmake inplace
    with fileinput.FileInput("../CMakeLists.txt", inplace=True) as file:
        for line in file:
            print(line.replace('set_target_properties(pyigl PROPERTIES COMPILE_FLAGS "-fvisibility=hidden -msse2")', 'set_target_properties(pyigl PROPERTIES COMPILE_FLAGS "-fvisibility=hidden -msse2 -H")'), end='')
    
    # Create tmpdir, run Cmake and make
    os.makedirs("../tmp_build", exist_ok=True)
    os.chdir("../tmp_build")
    os.system("cmake ..")
    deps = os.popen("make -j 4").read()
    print(len(deps))
    # Remove not included files
