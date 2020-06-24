import os
import sys
import yaml
import torch
import torch.multiprocessing as mp



from trainer import Trainer
from core.philly import ompi_size, ompi_local_size, ompi_rank, ompi_local_rank
from core.philly import get_master_ip, gpu_indices, ompi_universe_size
from core.philly import set_seed


 
            


def main_worker(gpu, ngpus_per_node, config):
  if 'local_rank' not in config:
    config['local_rank'] = config['global_rank'] = gpu
  if config['distributed']:
    torch.cuda.set_device(int(config['local_rank']))
    print('using GPU {} for training'.format(int(config['local_rank'])))
    torch.distributed.init_process_group(backend = 'nccl', 
      init_method = config['init_method'],
      world_size = config['world_size'], 
      rank = config['global_rank'],
      group_name='mtorch'
    )
  set_seed(config['seed'])
  trainer = Trainer(config)
  trainer.train()
    

if __name__ == '__main__':

  cfg_path = sys.argv[1]

  with open(cfg_path, 'r') as f:
    config = yaml.load(f)
  
  if config['single'] > 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config['single'])
    
    
  print('check if the gpu resource is well arranged on philly')
  assert ompi_size() == ompi_local_size() * ompi_universe_size()


  # setup distributed parallel training environments
  world_size = ompi_size()
  ngpus_per_node = torch.cuda.device_count()

  if world_size > 1:
    config['world_size'] = world_size
    config['init_method'] = 'tcp://' + get_master_ip() + config['port']
    config['distributed'] = True
    config['local_rank'] = ompi_local_rank()
    config['global_rank'] = ompi_rank()
    main_worker(0, 1, config)
  elif ngpus_per_node > 1:
    config['world_size'] = ngpus_per_node
    config['init_method'] = 'tcp://127.0.0.1:'+ config['port']
    config['distributed'] = True
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config))
  else:
    config['world_size'] = 1 
    config['distributed'] = False
    main_worker(0, 1, config)