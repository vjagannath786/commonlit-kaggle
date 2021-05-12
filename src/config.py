import transformers

DEVICE ="cpu"
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 8
NUM_WORKERS = 1
EPOCHS = 20
MAX_LEN = 205
LR = 3e-5
BERT_MODEL = 'bert-base-uncased'
ROBERTA_MODEL = 'roberta-base'
TRAIN_FILE = "../input/train_folds.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_MODEL, do_lower_case= True)
ROBERTA_TOKENIZER = transformers.RobertaTokenizer.from_pretrained(ROBERTA_MODEL, do_lower_case= True)


