import os
dress="./data/garbage_classify/train_data"
print(dress)
with open("train.txt","w") as f:
    for root,dirs,files in os.walk(dress):
        # root = root.replace(dress,'')
        #print(root, files)
        for file in files:
            if file.endswith(".jpg"):
                print("xx")
            else:
                #print("xx")
                f.write(os.path.join(root, file) + "\n")