from symspell import SymSpell, EditDistance, SuggestionItem
import json

ss = SymSpell(max_dictionary_edit_distance=4)
filename='freq_dict.json'

ss.load_words_with_freq_from_json_and_build_dictionary(filename)
ss.save_complete_model_as_json("test_completeModel.json")

with open('test_word_spellings.txt', 'r') as f:
    words=f.readlines()

actual_words=[]
with open(filename, 'r') as f:
    lines=json.load(f)
    actual_words=list(lines.keys())
    
tp=0
for term in range(len(words)):
    suggestion_list = ss.lookup(phrase=words[term], verbosity=1, max_edit_distance=4)
    if(len(suggestion_list)!=0):
        corrected_spelling=suggestion_list[0]._term
    else:
        corrected_spelling=words[term]
    
    if(actual_words[term]==corrected_spelling):
        tp+=1

print(tp)

