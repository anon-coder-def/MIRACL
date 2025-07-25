import argparse
def parse_noise_ratios(value):
    """
    Parse the noise_ratio argument, which can be a single float or a list of floats.
    Args:
        value (str): Input string value from the command line.
    Returns:
        list of float: Parsed noise ratios.
    """
    try:
        # Check if the input is a single float
        return [float(value)]
    except ValueError:
        # Otherwise, assume it's a list of floats
        try:
            return [float(x) for x in value.split(",")]
        except ValueError:
            raise argparse.ArgumentTypeError(
                "Noise ratio must be a float or a comma-separated list of floats."
            )
            
def args_parser():
    parser = argparse.ArgumentParser(description='arguments')

    parser.add_argument('--data_path', type=str, help='Path to the data',
                        default='data')
    parser.add_argument('--ehr_path', type=str, help='Path to the ehr data',
                        default='data/ehr')
    parser.add_argument('--imdb_path', type=str, help='Path to the cxr data',
                        default='data/imdb')

    parser.add_argument('--timestep', type=float, default=1.0, help="fixed timestep used in the dataset")
    parser.add_argument('--normalizer_state', type=str, default=None, help='Path to a state file of a normalizer. Leave none if you want to use one of the provided ones.')
    parser.add_argument('--resize', default=256, type=int, help='number of epochs to train')
    parser.add_argument('--crop', default=224, type=int, help='number of epochs to train')

    parser.add_argument('--task', type=str, default='phenotyping', help='in-hospital-mortality,length-of-stay,decompensation,phenotyping,readmission,diagnosis')
    parser.add_argument('--epochs', type=int, default=100, help='number of chunks to train')
    parser.add_argument('--corr_start', type=int, default=5, help='correction start time')
    parser.add_argument('--conf_start', type=int, default=5, help='confident learning start time')
    parser.add_argument('--device', type=str, default="cpu", help='cuda:number or cpu')
    parser.add_argument('--num_workers', type=int, default=8, help='num_workers for dataloader')
    parser.add_argument('--seed', type=int, nargs='+', default=[40], help='List of random seeds')

    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--model', type=str, default='our', help='our model or baselines')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--mean_loss_len', type=int,default=1,help='the length of mean loss')
    parser.add_argument('--gamma', type=float, default=0.5, help='the weight of current sample loss in mean_loss_sel method')
    parser.add_argument('--subset_ratio', type=float, default=0.2, help='subset of train and test set ratio')
    
    parser.add_argument('--nbins', type=lambda s: [int(x) for x in s.split(',')], default=[25], help='number of class')
    parser.add_argument('--standardization_choice', type=str, choices=['z-score', 'min-max'], default='z-score',
                        help='choose the method of standardization')
    parser.add_argument('--corr', action='store_true', help="Enable correction")
    
    parser.add_argument('--dataset', type=str, default='mimic3', help='mimic3,mimic4')
    
    parser.add_argument('--GlobalGMM', action='store_true', help="Enable GlobalGMM")
    parser.add_argument('--ClassGMM', action='store_true', help="Enable ClassGMM")
    parser.add_argument('--LocalGMM', action='store_true', help="Enable LocalGMM")
    parser.add_argument('--BMM', action='store_true', help="BMM")
    parser.add_argument('--contrast', action='store_true', help="Contrast or not")
    parser.add_argument('--contrast_subset', action='store_true', help="Contrast or not")
    
    
    parser.add_argument(
        '--noise_ratio', 
        type=parse_noise_ratios, 
        default=[0.5], 
        help="Noise ratio to apply. Accepts a single float or a comma-separated list of floats."
    )
    parser.add_argument('--noise_type', type=str, choices=['Symm', 'Asym','CCN' ,'Balanced'], default='Symm',
                        help="Type of label noise to add.")
    parser.add_argument('--subset', action='store_true', help="Enable subset training")
    parser.add_argument('--criterion',type =str,choices = ['BCE','Focal','MLLSC','SPLC','ASL','MLLSC','Hill','GCE'], help = 'Loss Function', default = 'BCE')
    parser.add_argument('--loss_coef', type=float, default=0, help='loss coefficient')
    parser.add_argument('--mem_coef', type=float, default=0.5, help='memorization coefficient')
    parser.add_argument('--rank_coef', type=float, default=0.5, help='rank coefficient')
    parser.add_argument('--score_coef', type=float, default=0, help='Score coefficient')
    parser.add_argument('--threshold_coef', type=float, default=0.02, help='threshold coefficient')
    parser.add_argument('--cont_coef', type=float, default=0.1, help='Patient CL coefficient')
    
    args = parser.parse_args()

    # args = argParser.parse_args()
    return parser
