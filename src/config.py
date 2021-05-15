import transformers

DEVICE ="cuda"
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 8
NUM_WORKERS = 8
EPOCHS = 80
MAX_LEN = 256
LR = 3e-5
BERT_MODEL = 'bert-base-uncased'
ROBERTA_MODEL = 'roberta-base'
TRAIN_FILE = "../../input/commonlit-folds/train_folds.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_MODEL, do_lower_case= True)
ROBERTA_TOKENIZER = transformers.RobertaTokenizer.from_pretrained(ROBERTA_MODEL, do_lower_case= True)


