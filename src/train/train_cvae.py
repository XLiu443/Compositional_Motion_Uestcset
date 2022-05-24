import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from src.train.trainer import train
from src.utils.tensors import collate
import src.utils.fixseed  # noqa

from src.parser.training import parser
from src.utils.get_model_and_data import get_model_and_data
import ipdb
from src.samplers.batch_sampler import BatchSampler 
import argparse

def do_epochs(model, datasets, parameters, optimizer, writer):
#    pdb.set_trace()
    ngpus_per_node = torch.cuda.device_count()
    print("ngpus_per_node", ngpus_per_node) 
    dataset = datasets["train"]
    train_sampler = DistributedSampler(dataset)
    train_iterator = DataLoader(dataset, sampler=train_sampler, batch_size=parameters["batch_size"]//ngpus_per_node, pin_memory=True,
                                shuffle=(train_sampler is None), num_workers=4, collate_fn=collate)

#    train_iterator = DataLoader(dataset, batch_size=parameters["batch_size"],
#                                shuffle=True, num_workers=8, collate_fn=collate)
 #   train_iterator = DataLoader(dataset, batch_size=parameters["batch_size"],
 #                               sampler=BatchSampler(dataset._train, dataset._actions, parameters["batch_size"], 4),
 #                               num_workers=8, collate_fn=collate_function)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[parameters["local_rank"]], output_device=parameters["local_rank"], find_unused_parameters=True)
#    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank)
    #ipdb.set_trace()
    logpath = os.path.join(parameters["folder"], "training.log")
    with open(logpath, "w") as logfile:
        for epoch in range(1, parameters["num_epochs"]+1):
        #    dict_loss = train(model, optimizer, train_iterator, model.device)
            dict_loss = train(model, optimizer, train_iterator, parameters["device"])
            for key in dict_loss.keys():
                dict_loss[key] /= len(train_iterator)
                #if parameters["local_rank"] == 0:
                writer.add_scalar(f"Loss/{key}", dict_loss[key], epoch)

            epochlog = f"Epoch {epoch}, train losses: {dict_loss}"
            if parameters["local_rank"] == 0:
                print(epochlog)
                print(epochlog, file=logfile)

            if ((epoch % parameters["snapshot"]) == 0) or (epoch == parameters["num_epochs"]):
                checkpoint_path = os.path.join(parameters["folder"],
                                               'checkpoint_{:04d}.pth.tar'.format(epoch))
                if parameters["local_rank"] == 0:
                    print('Saving checkpoint {}'.format(checkpoint_path))
            #    torch.save(model.state_dict(), checkpoint_path)
                    torch.save(model.module.state_dict(), checkpoint_path)
           # if parameters["local_rank"] == 0:
            writer.flush()


if __name__ == '__main__':
    # parse options
  #  parser2 = argparse.ArgumentParser()
  #  parser2.add_argument("--local_rank", type=int, default=-1)
  #  args2 = parser2.parse_args()
    torch.distributed.init_process_group(backend='nccl')
#    ipdb.set_trace()
    parameters = parser()
#    torch.distributed.init_process_group(backend='nccl')    
    # logging tensorboard
    #if parameters["local_rank"] == 0:
    writer = SummaryWriter(log_dir=parameters["folder"])
    model, datasets = get_model_and_data(parameters)

#    for name, value in model.mae_model.named_parameters():
#        value.requires_grad = False
#        print('name: {0},\t grad: {1}'.format(name, value.requires_grad))
    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=parameters["lr"])
#    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=parameters["lr"])
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    print("Training model..")
    do_epochs(model, datasets, parameters, optimizer, writer)
    #if parameters["local_rank"] == 0:
    writer.close()
