import os
pwd = os.getcwd()
tag = pwd.split('/')[1]
if tag=='lustre':
    DATA_DIR = '/lustre/home/acct-aemwx/aemwx-user1/wangyu/my/nlp-final/data'
elif tag=='dssg':
    DATA_DIR = '/dssg/home/acct-aemwx/aemwx-user1/wangyu/my/raw/data'
elif tag=='Users':
    DATA_DIR='/Users/zed/workspace/VSCode/SJTU/nlp-project/data'
elif tag=='root':
    DATA_DIR='/root/nlp-project/data'