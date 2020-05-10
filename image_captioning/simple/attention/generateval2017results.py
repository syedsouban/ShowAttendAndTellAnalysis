import json
import os
import nltk

val2017results=[]

files = [str(i[:-4]) for i in os.listdir("./val2017/")]

print(len(files))
fileslists=[]
fileslists.append(files[0:500])
fileslists.append(files[501:1000])
fileslists.append(files[1001:1500])
fileslists.append(files[1501:2000])
fileslists.append(files[2001:2500])
fileslists.append(files[2501:3000])
fileslists.append(files[3001:3500])
fileslists.append(files[3501:4000])
fileslists.append(files[4001:4500])
fileslists.append(files[4501:4999])


leftoutlist=[files[500],files[1000],files[1500],files[2000],files[2500],files[3000],files[3500],files[4000],files[4500],files[4999]]
#leftoutlists=[]


i=1
#for fileslist in leftoutlists:
val2017results=[]
for image in leftoutlist:
    print("count ",i)
    hypothesis=os.popen("python -W ignore caption.py --img=val2017/"+image+".jpg --model=BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar --word_map=WORDMAP_coco_5_cap_per_img_5_min_word_freq.json --beam_size=5").read()
    print(hypothesis)
    val2017results.append({"image_id":int(image),"caption":hypothesis})
    i=i+1        
with open('val2017resultslefout.json', 'w') as fp:
    json.dump(val2017results, fp, sort_keys=True, indent=4)
    
