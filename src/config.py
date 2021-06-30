import transformers
from transformers import AutoTokenizer

DEVICE ="cuda"
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 8
NUM_WORKERS = 8

EPOCHS = 5
MAX_LEN = 250
LR = 2e-5

BERT_MODEL = 'bert-base-uncased'
ROBERTA_MODEL = 'roberta-large'
TRAIN_FILE = "../../input/commonlit-folds/train_folds.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_MODEL, do_lower_case= True)
ROBERTA_TOKENIZER = AutoTokenizer.from_pretrained(ROBERTA_MODEL, do_lower_case= True)


