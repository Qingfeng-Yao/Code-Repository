import time
from torchnlp.word_to_vector import GloVe

from utils import parse_args, MINDDataset
from mylog import Logger
from nrms_models import NRMS

args = parse_args()
savepath = 'result/' + args.foldname + '/' + time.strftime('%m_%d-%H-%M-%S', time.localtime(time.time()))
log = Logger('root', savepath)
args.savepath = savepath
logger = log.getlog()
write_para = ''
for k, v in vars(args).items():
	write_para += '\n' + k + ' : ' + str(v)
logger.info('\n' + write_para + '\n')

data = MINDDataset(args)
if args.use_pretrained_embeddings:
	preemb = GloVe(name='840B', cache="data/word_vectors_cache")
else:
	preemb = None
model = NRMS(preemb,args,logger,data)
model.mtrain()