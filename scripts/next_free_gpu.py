import gpustat

def free(gpu):
    return len(gpu['processes']) == 0

gpus = gpustat.new_query()
print(next((idx for idx, gpu in enumerate(gpus) if free(gpu)), ""))
