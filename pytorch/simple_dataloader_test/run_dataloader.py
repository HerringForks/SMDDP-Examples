import torch
from torch.utils.data import Dataset, DataLoader
import os
import torch.distributed as dist


class SimpleDataset(Dataset):
    def __len__(self):
       return 1_000_000

    def __getitem__(self, item):
        return torch.tensor([item], dtype=torch.float32), torch.tensor([item], dtype=torch.float32)


def run_test():
    print('start')
    os.environ["SMDATAPARALLEL_LMC_ENABLE"] = "1"
    import smdistributed.dataparallel.torch.torch_smddp
    dist.init_process_group('smddp')
    print(f'done initializing process group')
    batch_data = next(iter(DataLoader(SimpleDataset(), batch_size=64, num_workers=4)))
    print(f'done extracting batch data in smddp')
    dist.barrier()
    dist.destroy_process_group()

if __name__ == '__main__':
    run_test()