import time
from torchnlp.word_to_vector import GloVe
import torch
import numpy as np
import random

from utils import parse_args, MINDDataset, HeyDataset
from mylog import Logger
from nrms_models import NRMS

args = parse_args()
savepath = 'result/' + args.modelname + '/' + args.dataset + '/' + time.strftime('%m_%d-%H-%M-%S', time.localtime(time.time()))
log = Logger('root', savepath)
args.savepath = savepath
logger = log.getlog()
write_para = ''
for k, v in vars(args).items():
	write_para += '\n' + k + ' : ' + str(v)
logger.info('\n' + write_para + '\n')

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True

if args.dataset == 'MIND':
	data = MINDDataset(args)
elif args.dataset == 'heybox':
	data = HeyDataset(args)

if args.pretrained_embeddings == 'glove':
	preemb = GloVe(name='840B', cache="data/word_vectors_cache")
else:
	preemb = None
	
if args.modelname == 'nrms':
	model = NRMS(preemb,args,logger,data)
model.mtrain()