import argparse
from engine import *
from models import *
from coco import *
from util import *
from post_process import *

basedir = os.path.abspath(os.path.dirname(__file__))


def par_option():
	parser = argparse.ArgumentParser(description='WILDCAT Training')
	parser.add_argument('data', metavar='DIR',
	                    help='path to dataset (e.g. data/')
	parser.add_argument('--image-size', '-i', default=448, type=int,
	                    metavar='N', help='image size (default: 224)')
	parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
	                    help='number of data loading workers (default: 4)')
	parser.add_argument('--epochs', default=100, type=int, metavar='N',
	                    help='number of total epochs to run')
	parser.add_argument('--epoch_step', default=[30], type=int, nargs='+',
	                    help='number of epochs to change learning rate')
	parser.add_argument('--device_ids', default=[0], type=int, nargs='+',
	                    help='number of epochs to change learning rate')
	parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
	                    help='manual epoch number (useful on restarts)')
	parser.add_argument('-b', '--batch-size', default=16, type=int,
	                    metavar='N', help='mini-batch size (default: 256)')
	parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
	                    metavar='LR', help='initial learning rate')
	parser.add_argument('--lrp', '--learning-rate-pretrained', default=0.1, type=float,
	                    metavar='LR', help='learning rate for pre-trained layers')
	parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
	                    help='momentum')
	parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
	                    metavar='W', help='weight decay (default: 1e-4)')
	parser.add_argument('--print-freq', '-p', default=0, type=int,
	                    metavar='N', help='print frequency (default: 10)')
	parser.add_argument('--resume', default='', type=str, metavar='PATH',
	                    help='path to latest checkpoint (default: none)')
	parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
	                    help='evaluate model on validation set')
	parser.add_argument('--word2vec_file', type=str, default='data/coco/coco_glove_word2vec.pkl')

	parser.add_argument('--test_set_amount', type=int, default=1)  
	parser.add_argument('--query_code_amount', type=int, default=496) 
	parser.add_argument('--testset_pkl_path', type=str, default='./data/coco/coco_test_set.pkl')  
	parser.add_argument("--query_pkl_path", type=str, default='./data/coco/coco_query_set.pkl')
	parser.add_argument("--hashcode_pool", type=str,
	                    default='./data/coco/coco_hashcode_pool.pkl')  
	parser.add_argument("--hashcode_pool_limit", type=int, default=25000)
	
	parser.add_argument('--DROPOUT_RATIO', type=float, default=0.1)  
	parser.add_argument('--CLASSIFIER_CHANNEL', type=str, default=2048)  
	parser.add_argument('--IMAGE_CHANNEL', type=int,
	                    default=2048) 
	parser.add_argument("--linear_intermediate", type=int,
	                    default=358)  
	parser.add_argument('--pooling_stride', type=int, default=358)  
	parser.add_argument("--threshold_p", type=float, default=0.15)  
	parser.add_argument("--threshold_tao", type=float, default=0.4)

	parser.add_argument("--gamma_weight", type=float, default=0.5)  
	parser.add_argument("--inter_channel", type=int, default=1024)  
	
	parser.add_argument("--IS_USE_MFB", action='store_true')  
	
	return parser


def main_coco():
	global args, best_prec1, use_gpu
	parser = par_option()
	args = parser.parse_args()
	num_classes = 80
	use_gpu = torch.cuda.is_available()
	state = {
		'batch_size': args.batch_size, 'image_size': args.image_size, 'max_epochs': args.epochs,
		'evaluate': args.evaluate,
		'resume': args.resume,
		'num_classes': num_classes,
		'test_set_amount': args.test_set_amount,
		'query_code_amount': args.query_code_amount,
		'testset_pkl_path': args.testset_pkl_path,
		'hashcode_pool': args.hashcode_pool,
		'query_pkl_path': args.query_pkl_path,
		'pooling_stride': args.pooling_stride,
		'linear_intermediate': args.linear_intermediate,
		'threshold_p': args.threshold_p,
		'threshold_tao': args.threshold_tao,
		'CLASSIFIER_CHANNEL': args.CLASSIFIER_CHANNEL,
		'IMAGE_CHANNEL': args.IMAGE_CHANNEL,
		'IS_USE_MFB': args.IS_USE_MFB,
		'gamma_weight': args.gamma_weight,
		'inter_channel': args.inter_channel,
	}
	state['difficult_examples'] = True
	state['save_model_path'] = 'checkpoint/coco/'
	state['workers'] = args.workers
	state['epoch_step'] = args.epoch_step
	state['lr'] = args.lr
	state["start_time"] = datetime.datetime.now()

	train_dataset = COCO2014(args.data, phase='train', inp_name=args.word2vec_file)
	val_dataset = COCO2014(args.data, phase='val', inp_name=args.word2vec_file)
	
	model = gcn_resnet101(args, num_classes=num_classes, t=0.4, adj_file='data/coco/coco_adj.pkl')
	
	criterion = nn.MultiLabelSoftMarginLoss()
	
	optimizer = torch.optim.SGD(model.get_config_optim(args.lr, args.lrp),
	                            lr=args.lr,
	                            momentum=args.momentum,
	                            weight_decay=args.weight_decay)
	
	if args.evaluate:
		state['evaluate'] = True
	engine = GCNMultiLabelMAPEngine(state)
	engine.learning(model, criterion, train_dataset, val_dataset, optimizer)
	engine.display_all_loss()


if __name__ == '__main__':
	main_coco()
	
