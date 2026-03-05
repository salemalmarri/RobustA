import argparse

parser = argparse.ArgumentParser(description='RobustA')
parser.add_argument('--modality', default='MIX2', 
                    help='the type of the input: AUDIO,RGB,FLOW, MIX, MIX2, MIX3, or MIX_ALL')
parser.add_argument('--rgb-list', default='list/rgb.list', help='list of rgb features')
parser.add_argument('--flow-list', default='list/flow.list', help='list of flow features')
parser.add_argument('--audio-list', default='list/audio.list', help='list of audio features')
parser.add_argument('--test-rgb-list', default='list/rgb_test.list', help='list of test rgb features')
parser.add_argument('--test-flow-list', default='list/flow_test.list', help='list of test flow features')
parser.add_argument('--test-audio-list', default='list/audio_test.list', help='list of test audio features')
parser.add_argument('--gt', default='list/gt.npy', help='file of ground truth')
parser.add_argument('--gpus', default=0, type=int, choices=[-1, 0, 1], help='gpus')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')

# New argument for weight decay
parser.add_argument('--weight-decay', type=float, default=0.000, 
                    help='weight decay for the optimizer (default: 0.0)')

# New argument for scheduler milestones
parser.add_argument('--scheduler-milestones', type=int, nargs='+', default=[10],
                    help='milestones for MultiStepLR scheduler, e.g. --scheduler-milestones 20 40')

parser.add_argument('--batch-size', type=int, default=640, help='batch size (default: 640)')
parser.add_argument('--workers', type=int, default=10, help='number of workers in dataloader')
parser.add_argument('--model-name', default='robusta', help='name to save model')
parser.add_argument('--pretrained-ckpt', default=None, help='ckpt for pretrained model')
parser.add_argument('--feature-size', type=int, default=2048, help='size of feature (default: 2048+128)')
parser.add_argument('--num-classes', type=int, default=1, help='number of classes')
parser.add_argument('--dataset-name', default='XD-Violence', help='dataset to train on')
parser.add_argument('--max-seqlen', type=int, default=200, help='max sequence length (default: 200)')
parser.add_argument('--max-epoch', type=int, default=50, help='max epoch (default: 50)')
parser.add_argument('--seed', type=int, default=2333, help='batch size (default: 2333)')


# New arguments for corruption
# parser.add_argument('--corrupt-modality', default=None, 
                    # help='which modality to corrupt: RGB, FLOW, AUDIO')
parser.add_argument('--corrupt-modality', nargs='+', default=[],
                    help='Modalities to corrupt: RGB, FLOW, AUDIO (can pass multiple)')
parser.add_argument('--corruption-percentage', type=float, default=0.0, 
                    help='fraction of samples to corrupt (0.0 to 1.0)')

# # New argument for single-branch feature summation
# parser.add_argument('--single-branch', action='store_true', 
                    # help='If set, use single-branch approach (pad and sum audio with RGB) instead of concatenation.')

parser.add_argument('--single-branch', 
                    action='store_true',  
                    default=True,
                    help='If set, use single-branch approach...')
                    
# New arguments for SBNet
parser.add_argument('--block-size', type=int, default=16, help='block size for SBNet')
parser.add_argument('--top-k-blocks', type=int, default=5, help='number of top blocks to keep')

# New argument to specify audio projection method
parser.add_argument('--audio-projection', default='lp', choices=['lp', 'mlp', 'pad', '2lp', '3lp'],
                    help='Projection method for audio: lp (linear projection) or mlp (multi-layer perceptron)')

parser.add_argument('--use-comet-run-name', action='store_true', 
                    help='If set, use Comet run name as part of the checkpoint name.')

# New argument for pre-corrupted test data
parser.add_argument('--test-corrupted-list', default='',
                    help='list of pre-corrupted test features (same indexing as the normal test list)')

parser.add_argument('--test_corrupted_audio_list', type=str, default='', 
                    help='Path to a text file listing pre-corrupted audio features for test mode.')

# ------------------------------------------------------------------------
# New arguments for alpha and beta
# ------------------------------------------------------------------------
parser.add_argument('--alpha', type=float, default=0.5,
                    help='Weight alpha for feature1 in weighted averaging (default: 0.5)')
parser.add_argument('--beta', type=float, default=0.5,
                    help='Weight beta for feature2 in weighted averaging (default: 0.5)')
parser.add_argument('--ell-multi', type=float, default=1.0,
                    help='stretch distribution')
                    
# ------------------------------------------------------------------------
# New arguments for alpha and beta
# ------------------------------------------------------------------------
parser.add_argument('--x0', type=float, default=-500,
                    help='')
# parser.add_argument('--c', type=float, default=0.001,
                    # help='')
parser.add_argument('--c', type=float, default=0.008,
                    help='')