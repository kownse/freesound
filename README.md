# Top 4% solution for Freesound General-purpose Audio tagging Challenge

This was my first time to handle sound classification problem because I missed a similar competition from kaggle.
It is sad that I didn't understand this is a research competition which means there was no medals althrough my 
solution ranked 19th. I think that was why not so many people joint it. 
But after all, I learnt to deal similar problems.

## Bad begin
It was natual to use RNN architectures since sound is a time serials signal, 
plus the data provided has different length which can only be handled in a whole by RNNs.
But I got no good results after trying LSTMs and GRUs no matter how deep they were.

Then I tried to trim the sounds to the first 5 seconds and input them into a 1D CNNs.
Then I even tried to transform the sound into mfcc 2d images and trained a 2d classifier
on them.
Still, none of above methods got really god result.

After search on internet and the kaggle forum, I got no good new ideas.

## Master's Insight
The blog from kaggle's global 1st [bestfitting](https://www.kaggle.com/bestfitting) who got
the 1st place in only 1 year gave me a good suggest: 
Read up all best solutions from the previous competitions.
This is a really gold suggestion which turns kaggle from a competitions platform into the 
best machine learning library in the world.
I found a very similar competition just a few month ago named [TensorFlow Speech Recognition Challenge](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge)
in which the winner turned the sounds into mel spectogram 2d images and then trained a image classifier on them.
![spectogram](https://i.imgur.com/P5S4wHB.png)

I followed this method and got a better score in leaderboard.
I got even better scores after I started training from pretrained weights from keras.
From this competitions, I really began to realize that the quality of the starting pretrained weights matters
a lot in all kinds of competitions.

## Stack Them All
The best solution from me is to train a big network contained as many models from many formats of 
sound including:
1. Image classifier on mel spectograms
2. Image classifier mfcc spectograms;
3. 1D CNNs on first 5 seconds;
4. GBTs on statistical features from sounds;

I really had fun in doing this since it is my first time to implement stack and it really worked.
