import json

captions={}

annot=json.loads(open('annotations/mscoco2017trainval.json').read())

for jo in annot:
    imid=str(jo['image_id'])
    
    zerostr=(12-len(imid))*'0'
    captions[zerostr+imid]=[]
for jo in annot:
    imid=str(jo['image_id'])
    zerostr=(12-len(imid))*'0'
    captions[zerostr+imid].append(jo['caption'])


with open('data.json', 'w') as fp:
    json.dump(captions, fp, sort_keys=True, indent=4)    

    
