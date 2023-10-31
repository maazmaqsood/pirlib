# from numba import cuda
import torch
from pirlib.iotypes import DirectoryPath, FilePath
from pirlib.pipeline import pipeline
from pirlib.task import task

@task
def get_version(dataset: DirectoryPath) -> DirectoryPath:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device.type)
    outdir = task.context().output
    file_name = outdir / "file.txt"
    print(device)
    with open(file_name, "w") as f:
        f.write(str(device))
    return outdir


@pipeline
def pipeline(dataset: DirectoryPath) -> DirectoryPath:
    data = get_version(dataset)
    return data
