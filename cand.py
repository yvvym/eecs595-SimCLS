from transformers import BartTokenizer, BartForConditionalGeneration

bart_path = 'facebook/bart-large-cnn'
device = 'cpu'

model = BartForConditionalGeneration.from_pretrained(bart_path)
model.to(device)
tokenizer = BartTokenizer.from_pretrained(bart_path)
# article = ["club tijuana star juan arango conjured memories luis suarez in his team 's 4-3 defeat by monterrey in the mexican league - but it was not through prodigious scoring .", "the venezuelan icon arango sank his teeth into the shoulder of jesus zavela as his temper flared in the defeat .", ".", "he was not booked by the referee but could face a heavy retrospective ban .", ".", "juan arango -lrb- left -rrb- bites the shoulder of opponent jesus zavela in a moment of madness .", "zavala holds his shoulder after being bitten by arango , in the game zavala 's side won 4-3 in mexico .", "zavala shows the referee the mark on his shoulder after being bittern by arango .", "arango -lrb- right -rrb- earlier scored a magnificent free kick to bring his club tijuana team level against monterrey .", "arango had earlier curled in a magnificent free kick for his team to bring them level after falling 2-0 down early on in the encounter .", "but the 34-year-old overshadowed his goal with the bite as television cameras picked up the moment of madness .", "arango spent 10 years playing in europe , spending five seasons each at real mallorca in spain and borussia monchengladbach in germany .", "he has made 121 appearances for venezuela .", "."]
# slines = [' '.join(article)]
slines = ["today i will come to cinema with alice, will you come?"]
# max_length = 140
# min_length = 55
max_length = 50
min_length = 5
print(slines)
dct = tokenizer.batch_encode_plus(slines, max_length=1024, return_tensors="pt", pad_to_max_length=True, truncation=True)
print(dct["input_ids"].shape)
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

i = 1
for abs in abstract_list:
    print(i, abs)
    i += 1


