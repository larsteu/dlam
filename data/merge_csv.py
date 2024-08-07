import os
import sys

if __name__ == "__main__":
    # get all the csv from the two given folders
    folder1 = sys.argv[1]
    folder2 = sys.argv[2]

    # get all the csv files from the folders
    files1 = [f for f in os.listdir(folder1) if f.endswith(".csv")]
    files2 = [f for f in os.listdir(folder2) if f.endswith(".csv")]

    # append the csv from files2 to files1 (the ones with the same name)
    # delete the first line, because it's the header
    for file1 in files1:
        for file2 in files2:
            if file1 == file2:
                with open(f"{folder1}/{file1}", "a") as f1:
                    with open(f"{folder2}/{file2}", "r") as f2:
                        # skip the first line
                        f2.readline()
                        f1.write(f2.read())
                break

    print("done")
