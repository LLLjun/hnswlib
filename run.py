import os

def main():
    stage = "both"
    Datasets = ["sift", "spacev"]
    datasize = 100
    Sub_graph = [4, 8]
    # Sub_graph = [1]

    for dataset in Datasets:
        for sg in Sub_graph:
            command = "./main " + stage + " " + dataset + " " + str(datasize) + " " + str(sg)
            os.system("cd build && make main && " + command)

main()