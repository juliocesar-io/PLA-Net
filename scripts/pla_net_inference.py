import torch
import numpy as np
import pandas as pd
import time
from torch_geometric.data import DataLoader
from model.model_concatenation import PLANet
from utils.args import ArgsInit
from utils.model import get_dataset_inference, test_gcn


def main(args):

    if args.use_gpu:
        device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device('cpu')
    
    #Numpy and torch seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)
    print('%s' % args)
    
    
    data_inference = pd.read_csv(
        args.input_file_smiles, 
        names=["Smiles"],
        header=0
    )
 
    print("Data Inference: ", data_inference)

    data_target = pd.read_csv(
        args.target_list, names=["Fasta", "Target", "Label"]
    )
    data_target = data_target[data_target.Target == args.target]

    print("Data Target: ", data_target)

    test = get_dataset_inference(
        data_inference,
        use_prot=args.use_prot,
        target=data_target,
        args=args,
        advs=False,
        saliency=False,
    )
    
    print("Dataset inference: ", test)
    
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    model = PLANet(args).to(device)
    
    print(model)


    print('Model inference in: {}'.format(args.inference_path))
    start_time = time.time()

    #Load pre-trained molecule model

    print('Evaluating...')
    test_gcn(model, device, test_loader, args)


    end_time = time.time()
    total_time = end_time - start_time
    print('Total time: {}'.format(time.strftime('%H:%M:%S', time.gmtime(total_time))))


if __name__ == "__main__":
    args = ArgsInit().args
    # Default args for inference
    
    args.nclasses = 2
    args.batch_size = 10
    args.use_prot = True
    args.freeze_molecule = True
    args.conv_encode_edge = True
    args.learn_t = True
    args.binary = True
    
    main(args)
