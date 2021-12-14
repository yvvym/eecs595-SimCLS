from os import sep
import pandas as pd
import numpy as np
import argparse
import csv
import pickle
from transformers import BartTokenizer, BartForConditionalGeneration
import torch

def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="load")
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--max_length", type=int, required=True)
    parser.add_argument("--min_length", type=int, required=True)
    args = parser.parse_args()
    # file = args.data + "_data/" + args.split + ".csv"
    data = load_obj(args.data + "_" + args.split)

    f_out = open(args.data + "/" + args.split + ".out", "w")
    f_out_tok = open(args.data + "/" + args.split + ".out.tokenized", "w")
    f_src = open(args.data + "/" + args.split + ".source", "w")
    f_src_tok = open(args.data + "/" + args.split + ".source.tokenized", "w")
    f_tgt = open(args.data + "/" + args.split + ".target", "w")
    f_tgt_tok = open(args.data + "/" + args.split + ".target.tokenized", "w")

    bart_path = 'facebook/bart-large-cnn'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = BartForConditionalGeneration.from_pretrained(bart_path)
    model.to(device)
    tokenizer = BartTokenizer.from_pretrained(bart_path)

    for count, text in enumerate(data):
        if count / 100 == 0:
            print(count)
        article = text["article"]
        # print(article)
        summary = text["summary"]
        # print(summary)
        # break

        f_src.write(article + "\n")
        f_src_tok.write(article + "\n")
        f_tgt.write(summary + "\n")
        f_tgt_tok.write(summary + "\n")

        slines = [article]
        dct = tokenizer.batch_encode_plus(slines, max_length=1024, return_tensors="pt", pad_to_max_length=True, truncation=True)
        summaries = model.generate(
            input_ids=dct["input_ids"].to(device),
            attention_mask=dct["attention_mask"].to(device),
            num_return_sequences=16, num_beam_groups=16, diversity_penalty=1.0, num_beams=16,
            max_length=args.max_length + 2,  # +2 from original because we start at step=1 and stop before max_length
            min_length=args.min_length + 1,  # +1 from original because we start at step=1
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
    # dataset = np.loadtxt
    # with open(file, newline="\n") as f:
    #     reader = csv.reader(f)
    #     for row in reader:
    #         print(row)
    # print(args.train_file_name, args.test_file_name, args.val_file_name)