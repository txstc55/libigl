import os
import argparse

parser = argparse.ArgumentParser(description='Process demo type and file name')
parser.add_argument("--d", default= 1, help="Demo type, from 1 to 3")
parser.add_argument("--f", default= "", help="input mesh name, has to be in tutorial/data folder already")
args = parser.parse_args()
file = args.f
demo_type = args.d

if (file!=""):
    file = " -f "+file

demo_type = " -d "+ str(demo_type) 
## check if build exists
if not os.path.isdir("build"):
    print("Creating build folder now")
    os.system("mkdir build")
    os.system("cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j && cd ..")
else:
    print("Build exists, rebuilding")
    os.system("cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j && cd ..")

## check if result data folder exists
if not os.path.isdir("result_datas"):
    print("Creating result datas folder now")
    os.system("mkdir result_datas")

## do eigen tests
os.system("cd build && ./tutorial/709_SLIM_bin -m 0" + demo_type + file +" && cd ..")
os.system("mv build/result.txt result_datas/result_eigen.txt")



## do cached tests
os.system("cd build && ./tutorial/709_SLIM_bin -m 1" + demo_type + file +" && cd ..")
os.system("mv build/result.txt result_datas/result_cached.txt")



## do mkl tests
os.system("cd build && ./tutorial/709_SLIM_bin -m 2" + demo_type + file +" && cd ..")
os.system("mv build/result.txt result_datas/result_mkl.txt")


## do numeric 1 tests
os.system("cd build && ./tutorial/709_SLIM_bin -m 3" + demo_type + file +" && cd ..")
os.system("mv build/result.txt result_datas/result_numeric1.txt")

## do numeric 2 tests
os.system("cd build && ./tutorial/709_SLIM_bin -m 4" + demo_type + file +" && cd ..")
os.system("mv build/result.txt result_datas/result_numeric2.txt")


eigen_data = {}
eigen_data["name"] = "EIGEN"
eigen_data["COMPUTE"] = 0
eigen_data["SOLVE"] = 0
eigen_data["ASSEMBLE"] = 0
f = open("result_datas/result_eigen.txt")
count = 0
for line in f:
    if not line.startswith("START"):
        count = count%11
        if count == 0 or count == 1 or count == 2 or count == 3 or count == 5 or count == 6 or count == 8:
            eigen_data["ASSEMBLE"] += float(line.split(": ")[1])
        elif count == 4 or count == 7:
            eigen_data["COMPUTE"] += float(line.split(": ")[1])
        else:
            eigen_data["SOLVE"] += float(line.split(": ")[1])
        count+=1




cached_data = {}
cached_data["name"] = "CACHED"
cached_data["COMPUTE"] = 0
cached_data["SOLVE"] = 0
cached_data["ASSEMBLE"] = 0
f = open("result_datas/result_cached.txt")
count = 0
for line in f:
    if not line.startswith("START"):
        count = count%11
        if count == 0 or count == 1 or count == 2 or count == 5 or count == 6 or count == 8:
            cached_data["ASSEMBLE"] += float(line.split(": ")[1])
        elif count == 3 or count == 4 or count == 7:
            cached_data["COMPUTE"] += float(line.split(": ")[1])
        else:
            cached_data["SOLVE"] += float(line.split(": ")[1])
        count+=1




mkl_data = {}
mkl_data["name"] = "MKL"
mkl_data["COMPUTE"] = 0
mkl_data["SOLVE"] = 0
mkl_data["ASSEMBLE"] = 0
f = open("result_datas/result_mkl.txt")
count = 0
for line in f:
    if not line.startswith("START"):
        count = count%14
        if (count >= 0 and count<=8) or count == 11:
            mkl_data["ASSEMBLE"] += float(line.split(": ")[1])
        elif count == 9 or count == 10:
            mkl_data["COMPUTE"] += float(line.split(": ")[1])
        else:
            mkl_data["SOLVE"] += float(line.split(": ")[1])
        count+=1




numeric1_data = {}
numeric1_data["name"] = "NUMERIC1"
numeric1_data["COMPUTE"] = 0
numeric1_data["SOLVE"] = 0
numeric1_data["ASSEMBLE"] = 0
f = open("result_datas/result_numeric1.txt")
count = 0
for line in f:
    if not line.startswith("START"):
        count = count%9
        if (count >= 0 and count<=5):
            numeric1_data["ASSEMBLE"] += float(line.split(": ")[1])
        elif count == 6:
            numeric1_data["COMPUTE"] += float(line.split(": ")[1])
        else:
            numeric1_data["SOLVE"] += float(line.split(": ")[1])
        count+=1







numeric2_data = {}
numeric2_data["name"] = "NUMERIC2"
numeric2_data["COMPUTE"] = 0
numeric2_data["SOLVE"] = 0
numeric2_data["ASSEMBLE"] = 0
f = open("result_datas/result_numeric2.txt")
count = 0
for line in f:
    if not line.startswith("START"):
        count = count%10
        if (count >= 0 and count<=3) or count == 5:
            numeric2_data["ASSEMBLE"] += float(line.split(": ")[1])
        elif count == 4 or count == 6 or count == 7:
            numeric2_data["COMPUTE"] += float(line.split(": ")[1])
        else:
            numeric2_data["SOLVE"] += float(line.split(": ")[1])
        count+=1



import json
with open('result_datas/all_result.json', 'w') as j:
    json.dump([eigen_data,cached_data,mkl_data,numeric1_data,numeric2_data],j)