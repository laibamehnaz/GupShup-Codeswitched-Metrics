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
