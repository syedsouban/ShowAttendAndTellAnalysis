import os
import subprocess
import nltk
import json

from itertools import islice

def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))


#output = subprocess.check_output("python caption.py --img='img/bleutest1.jpg' --model='BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar' --word_map='WORDMAP_coco_5_cap_per_img_5_min_word_freq.json' --beam_size=5", shell=True)
#proc = subprocess.Popen(["python caption.py --img=img/bleutest1.jpg --model='BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar --word_map=WORDMAP_coco_5_cap_per_img_5_min_word_freq.json --beam_size=5"], stdout=subprocess.PIPE, shell=True)
#(out, err) = proc.communicate()
#print("program output:", output)

#sent=os.popen("python -W ignore caption.py --img=img/bleutest1.jpg --model=BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar --word_map=WORDMAP_coco_5_cap_per_img_5_min_word_freq.json --beam_size=5").read()
#subprocess.call("caption.py --img='img/bleutest1.jpg' --model='BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar' --word_map='WORDMAP_coco_5_cap_per_img_5_min_word_freq.json' --beam_size=5", shell=True)
#sent=os.system("python caption.py --img='img/bleutest1.jpg' --model='BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar' --word_map='WORDMAP_coco_5_cap_per_img_5_min_word_freq.json' --beam_size=5")
#hypothesis=sent.split(" ")
#hypothesis[-1]=hypothesis[-1][:-1]
#print(hypothesis)

captions={}

val2017results=[]


annot=json.loads(open('data.json').read())


n_imgs = take(1000,annot.items())   



sumbleu1=0
sumbleu2=0
sumbleu3=0
sumbleu4=0
for index in range(0,1000):
    hypothesis=os.popen("python -W ignore caption.py --img=val2017/"+n_imgs[index][0]+".jpg --model=BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar --word_map=WORDMAP_coco_5_cap_per_img_5_min_word_freq.json --beam_size=5").read()


    hypothesis=hypothesis.split(" ")

    for i in range(0,len(hypothesis)):
        hypothesis[i]=hypothesis[i].lower()
        hypothesis[i].replace('.','')
        hypothesis[i].replace('\n','')
    

    strlist=n_imgs[index][1]


    reference=[]
    for str in strlist:
        str=str.lower()
        str=str.replace('\n','')
        str=str.replace('.','')
        reference.append(str.split(" "))
    
    print(reference)
    print(hypothesis)
    
    bleu1=nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weights=(1, 0, 0, 0))
    bleu2=nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weights=(0.5, 0.5, 0, 0))
    bleu3=nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weights=(0.33, 0.33, 0.33, 0))
    bleu4=nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weights=(0.25, 0.25, 0.25, 0.25))
    print('Cumulative 4-gram: %f' % bleu1)
    print('Cumulative 4-gram: %f' % bleu2)
    print('Cumulative 4-gram: %f' % bleu3)
    print('Cumulative 4-gram: %f' % bleu4)

    sumbleu1+=bleu1
    sumbleu2+=bleu2
    sumbleu3+=bleu3
    sumbleu4+=bleu4

    print('avg bleu1=%f' % (sumbleu1/(index+1)))
    print('avg bleu2=%f' % (sumbleu2/(index+1)))
    print('avg bleu3=%f' % (sumbleu3/(index+1)))
    print('avg bleu4=%f' % (sumbleu4/(index+1)))


print('avg bleu1=%f' % (sumbleu1/1000))
print('avg bleu2=%f' % (sumbleu2/1000))
print('avg bleu3=%f' % (sumbleu3/1000))
print('avg bleu4=%f' % (sumbleu4/1000))
    
