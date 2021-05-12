import transformers

DEVICE ="cuda"
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 8
NUM_WORKERS = 8
EPOCHS = 20
MAX_LEN = 300
LR = 3e-5
BERT_MODEL = 'bert-base-uncased'
TRAIN_FILE = "../input/train_folds.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_MODEL, do_lower_case= True)

