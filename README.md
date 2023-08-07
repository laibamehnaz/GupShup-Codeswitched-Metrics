# GupShup dataset statistics and code-mixed metrics
This repository contains the code for calculating different statistics and code-mixed metrics for the [GupShup](https://aclanthology.org/2021.emnlp-main.499/) dataset. The code for training the models can be found [here](https://github.com/midas-research/gupshup).

GupShup is a dataset that contains ~6800 conversations. These conversations are present in code-mixed Hindi and English. Along with these conversations, there is an English summary of that conversation present as well, per conversation. GupShup is the largest code-mixed Hindi English conversation summarization dataset.

An example of a conversation from the dataset is:

> **Leon**: kya tujeh abhi tak naukari nahi mili? </br>
> **Arthur**: nahi bro, abhi bhi unemployed :D </br>
> **Leon**: hahaha, LIVING LIFE </br>
> **Arthur**: mujeh yeh bahot acha lagta hai, dopahar ko jagata hoon, sports dekhta hoon - ek aadmi ko aur kya chahiye? </br>
> **Leon**: a paycheck? ;) </br>
> **Arthur**: mean mat bano ... </br>
> **Leon**: but seriously, mere dosth ke company mein ek junior project manager offer hai, tujeh interest hai? </br>
> **Arthur**: sure thing, tere pass details hai? </br>
> **Leon**: <file_photo> </br>
> </br>
> (**English Summary**): Arthur is still unemployed. Leon sends him a job offer for junior project manager position. Arthur is interested.


To use metrics.py, run the following command:
```
python metrics.py --data_dir /data/conversations_train_for_the_library.json \
  --hindi_tags /data/hindi_tags.txt \
  --english_tags /data/english_tags.txt \
  --ne_tags /data/ne_tags.txt \
  --hi_Latn_wordbigrams /data/hi-Latn-wordbigrams.txt \
  --output_dir /output/ \
  --n 200
```

```conversations_train_for_the_library.json``` This contains the Gupshup conversations and summaries with unique ids. </br>
```hindi_tags.txt``` This contains the words/tokens that are tagged as Hindi in the dataset. (The tags are given to words/tokens depending upon what conversation and what utterance they belong to.) </br>
```english_tags.txt``` This contains the words/tokens that are tagged as English in the dataset. (The tags are given to words/tokens depending upon what conversation and what utterance they belong to.) </br>
```ne_tags.txt``` This contains the words/tokens that are tagged as Named Entities in the dataset.  </br>
```hi-Latn-wordbigrams.txt```  Is a huge and comprehensive lexicon of most common Hindi bigrams, written in Latin script, and not Devanagari. </br>
```n``` This is used to indicate the number of samples to be used to calculate the statistics for. Use -1 to use the entire dataset.

The output of the script shows the following statistics:
```
------------ Dataset statistics ------------
Total number of conversations: 200
Total number of utterances: 2224
Total number of code-mixed utterances: 747
Total number of Hindi utterances: 980
Total number of English utterances: 228

Average length of utterances in the corpus: 9
Average number of utterances per conversation: 11
Average number of code-mixed utterances per conversation: 3
Percentage of code-mixed utterances in the corpus: 33.59%
Percentage of Hindi utterances in the corpus: 44.06%
Percentage of English utterances in the corpus: 10.25%

Total vocabulary size: 4012
Total English vocabulary size: 349
Total code-mixed English vocabulary size: 921
Total Hindi vocabulary size: 1677
Total ner vocabulary size: 343
Total other vocabulary size: 1809

Total number of code-mixed sentences: 786
Total number of sentences with Hindi matrix: 631
Total number of sentences with English matrix: 155

Number of sentences with English insertions: 583
Number of sentences with Hindi insertions: 152

Number of sentences with English alternations: 47
Number of sentences with Hindi alternations: 31

Number of sentences with single word insertion(English) : 486
Number of sentences with multi word insertion(English) : 144

Number of sentences with single word insertion(Hindi) : 71
Number of sentences with multi word insertion(Hindi) : 83

**** Distribution of number of Hindi insertions in English matrix sentences: ****
Number of sentences with 2 insertions: 44
Number of sentences with 1 insertions: 45
Number of sentences with 3 insertions: 27
Number of sentences with 6 insertions: 6
Number of sentences with 4 insertions: 16
Number of sentences with 0 insertions: 7
Number of sentences with 5 insertions: 7
Number of sentences with 9 insertions: 1
Number of sentences with 13 insertions: 1
Number of sentences with 11 insertions: 1
**** Distribution of number of English insertions in Hindi matrix sentences: ****
Number of sentences with 1 insertions: 356
Number of sentences with 2 insertions: 154
Number of sentences with 4 insertions: 32
Number of sentences with 5 insertions: 6
Number of sentences with 3 insertions: 53
Number of sentences with 7 insertions: 3
Number of sentences with 6 insertions: 8
Number of sentences with 0 insertions: 3
Number of sentences with 9 insertions: 3
Number of sentences with 10 insertions: 2
Number of sentences with 19 insertions: 2
Number of sentences with 11 insertions: 1
Number of sentences with 8 insertions: 6
Number of sentences with 14 insertions: 1
Number of sentences with 12 insertions: 1

K non-native(english) words distribution in the code-mixed corpora:
k, sentences:
(0, 0)
(1, 439)
(2, 175)
(3, 69)
(4, 42)
(5, 11)
(6, 12)
Figure(640x480)
Figure for k non-native words distribution is saved in :/output/

```
The above is the output for ```n=200```. The script also saves the bar graph of k-non-native foreign words into the output folder. For the above code, the following graph is generated and saved in the output folder.


<img src="/k_non_native_words.png" width="400" height="350">

