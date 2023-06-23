import argparse


def pars_args():
    parser = argparse.ArgumentParser(description='IR Model')

    # Where to find data
    parser.add_argument('--data_dir', type=str, default=r'F:\University\Term8\Information Retrieval\Project\src\data',
                        help='The dataset directory.')
    parser.add_argument('--cache_dir', type=str, default=r'F:\University\Term8\Information Retrieval\Project\src\out',
                        help='The processed dataset directory')

    # Important settings
    parser.add_argument('--model', type=str, default='Transformer', help='model structure')
    parser.add_argument('--restore_model', type=str, default='None',
                        help='Restore model for further training. [bestmodel//None]')

    # Where to save output
    parser.add_argument('--save_root', type=str, default='save/', help='Root directory for all model.')
    parser.add_argument('--log_root', type=str, default='log/', help='Root directory for all logging.')

    # Hyper Parameters
    parser.add_argument('--seed', type=int, default=666, help='set the random seed [default: 666]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID to use. [default: 0]')
    parser.add_argument('--cuda', action='store_true', default=False, help='GPU or CPU [default: False]')
    parser.add_argument('--vocab_size', type=int, default=50000, help='Size of vocabulary. [default: 50000]')
    parser.add_argument('--n_epochs', type=int, default=20, help='Number of epochs [default: 20]')
    parser.add_argument('--batch_size', type=int, default=32, help='Mini batch size [default: 32]')
    parser.add_argument('--n_iter', type=int, default=1, help='iteration hop [default: 1]')

    # Training
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--lr_descent', action='store_true', default=False, help='learning rate descent')
    parser.add_argument('--grad_clip', action='store_true', default=False, help='for gradient clipping')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='for gradient clipping max gradient normalization')

    # Summary Length
    parser.add_argument('-m', type=int, default=3, help='decode summary length')

    args = parser.parse_args()

    return args
