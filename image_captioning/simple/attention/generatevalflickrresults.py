import json
import os
import nltk

flickr8kresults=[]


files=[]

files = [i[:-4] for i in os.listdir("./Flicker8k_Dataset/")]

print(len(files))
fileslists=[]
fileslists.append(files[0:501])
fileslists.append(files[501:1001])
fileslists.append(files[1001:1501])
fileslists.append(files[1501:2001])
fileslists.append(files[2001:2501])
fileslists.append(files[2501:3001])
fileslists.append(files[3001:3501])
fileslists.append(files[3501:4001])
fileslists.append(files[4001:4501])
fileslists.append(files[4501:5000])

i=1
for fileslist in fileslists:
    flickr8kresults=[]
    for image in fileslist:
        print("count ",i)
        print(image)
        hypothesis=os.popen("python -W ignore caption.py --img=Flicker8k_Dataset/"+image+".jpg --model=BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar --word_map=WORDMAP_coco_5_cap_per_img_5_min_word_freq.json --beam_size=5").read()
        print(hypothesis[:-1])
        flickr8kresults.append({"image_id":image,"caption":hypothesis[:-1]})
        i=i+1        
    with open('flickr8kresults+'+str(i)+'.json', 'w') as fp:
        json.dump(flickr8kresults, fp, sort_keys=True, indent=4)
