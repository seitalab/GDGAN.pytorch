def create_subparser(subparsers):
    parser = subparsers.add_parser('GDGAN')

    parser.add_argument('--dataset', type=str, default='chest-xray',
                        choices=['mnist',
                                 'chest-xray'],
                        help='name of dataset')
    parser.add_argument('--root_path', type=str,
                        default="./data/nih-chest-xrays/images")
    parser.add_argument('--label_path', type=str,
                        default="./data/nih-chest-xrays/"
                                "labels/train_val_list2.txt")
    parser.add_argument('--batch_size', type=int, default=32,
                        help='size of batch')
    parser.add_argument('--image_size', type=int, default=-1,
                        help='size of image. -1 indicates default image size')
    parser.add_argument('--nc', type=int, default=1)
    parser.add_argument('--nz', type=int, default=128)
    parser.add_argument('--n_critic', type=int, default=1)
    parser.add_argument('--lambda_', type=int, default=10)
    parser.add_argument('--lrG', type=float, default=0.0002)
    parser.add_argument('--lrD', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)

    """
    loss config
    as for lambdas, please refer to README.md
    """
    parser.add_argument('--mse_loss', action='store_true')

    parser.add_argument('--lambda_G1_1', type=float, default=1)

    parser.add_argument('--lambda_D1_1', type=float, default=1)
    parser.add_argument('--lambda_D1_2', type=float, default=1)

    parser.add_argument('--lambda_G2_1', type=float, default=1)
    parser.add_argument('--lambda_G2_2', type=float, default=1)
    parser.add_argument('--lambda_G2_3', type=float, default=1)

    parser.add_argument('--lambda_D2_1', type=float, default=1)
    parser.add_argument('--lambda_D2_2', type=float, default=1)
    parser.add_argument('--lambda_D2_3', type=float, default=1)

    """
    for G1 and D1
    """
    parser.add_argument('--epoch1', type=int, default=20, help='# of epoch1')
    parser.add_argument('--y1_indices', type=str, default='',
                        help='ex: 0,1,2')
    parser.add_argument('--G1_path', type=str, default="",
                        help='model path of generator1')
    parser.add_argument('--D1_path', type=str, default="",
                        help='model path of discriminator1')

    """
    for G2 and D2
    """
    parser.add_argument('--epoch2', type=int, default=20, help='# of epoch')
    parser.add_argument('--y2_indices', type=str, default='',
                        help='ex: 0,1,2')


def validate_args(args):
    if args.dataset in ['mnist']:
        args.nc = 1
        if args.image_size == -1:
            args.image_size = 28
    elif args.dataset in ['chest-xray']:
        args.nc = 1
        if args.image_size == -1:
            args.image_size = 1024

    args.y1_indices = [int(i) for i in args.y1_indices.split(',')]
    args.y2_indices = [int(i) for i in args.y2_indices.split(',')]

    return args
