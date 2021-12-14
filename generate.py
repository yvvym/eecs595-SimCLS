from os import spawnl
import tensorflow_datasets as tfds
from transformers import BartTokenizer, BartForConditionalGeneration
import torch

bart_path = 'facebook/bart-large-cnn'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

s = 'test'
path = 'gigaword/' + s + '/'
model = BartForConditionalGeneration.from_pretrained(bart_path)
model.to(device)
tokenizer = BartTokenizer.from_pretrained(bart_path)
ds = tfds.load('gigaword',split=s)
max_length = 50
min_length = 5
data = {}

f_out = open(path + s + ".out", "w")
f_out_tok = open(path + s + ".out.tokenized", "w")
f_src = open(path + s + ".source", "w")
f_src_tok = open(path + s + ".source.tokenized", "w")
f_tgt = open(path + s + ".target", "w")
f_tgt_tok = open(path + s + ".target.tokenized", "w")
for count, text in enumerate(ds):
    if count / 100 == 0:
        print(count)
    article = text['document'].numpy().decode('utf-8')
    target = text['summary'].numpy().decode('utf-8')

    f_src.write(article + "\n")
    f_src_tok.write(article + "\n")
    f_tgt.write(target + "\n")
    f_tgt_tok.write(target + "\n")

    slines = [article]
    dct = tokenizer.batch_encode_plus(slines, max_length=1024, return_tensors="pt", pad_to_max_length=True, truncation=True)
    summaries = model.generate(
        input_ids=dct["input_ids"].to(device),
        attention_mask=dct["attention_mask"].to(device),
        num_return_sequences=16, num_beam_groups=16, diversity_penalty=1.0, num_beams=16,
        max_length=max_length + 2,  # +2 from original because we start at step=1 and stop before max_length
        min_length=min_length + 1,  # +1 from original because we start at step=1
        no_repeat_ngram_size=3,
        length_penalty=2.0,
        early_stopping=True,
    )
    abstract_list = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]

    for abs in abstract_list:
        f_out.write(abs + "\n")
        f_out_tok.write(abs + "\n")

f_out.close()
f_out_tok.close()
f_src.close()
f_src_tok.close()
f_tgt.close()
f_tgt_tok.close()