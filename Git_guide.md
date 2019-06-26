# Git guide

[Nice site](https://dont-be-afraid-to-commit.readthedocs.io/en/latest/git/commandlinegit.html)

## Open a new repository

opening a new git repo can be a pain, here I try to make it easier by adding one more guide to the millions around the web.

### 1. Open repository in git lab

Go to Gitlab and open a repo there, copy the repo's url, for example

```c
http://gitlab-srv/yotampe/checkerboards
```

### 2. Create and add folder to repo

create folder, cd there, **create a .gitignore**  in order for there not to be problems later, then:

```bash
# Initialize directory as a repository:
git init

# Let git know where is it's repository
git remote add origin http://gitlab-srv/yotampe/reponame.git

# Adds the files in the local repository and stages them for commit.
# Note, this have to be done after a .gitignore as created!

# create a git ignore 9I have one in /Code

# run a script that puts all files larger than 2MB in the .gitignore
find . -size +2M | sed 's|^\./||g' | cat >> .gitignore

git add .

# Commit changes
git commit -m "First commit"

# Pull whatever is in the repository
git pull origin master --allow-unrelated-histories

# Push whatever is in the folder
git push -u origin master
```

### Make git add forget all files added

```bash
git rm --cached
```



### Remove a commit

```bash
git log --pretty=oneline --abbrev-commit

# get: 
# 46cd867 Changed with mistake
# d9f1cf5 Changed again
# 105fd3d Changed content
# df33c8a First commit

git reset --hard HEAD~CommitsToRemove

# e.g to remove d9f1cf5 and 46cd867 reorting to 105fd3d CommitsToRemove=2
```

### Restore a removed commit

```bash
git reflog

# get 
# 52cf680 HEAD@{0}: commit: '1'
# 41720f8 HEAD@{1}: commit: 1
# de5302d HEAD@{2}: reset: moving to HEAD~1
# 84fc7ff HEAD@{3}: commit: before changing for sliding windows
# de5302d HEAD@{4}: reset: moving to HEAD~3
# f0b248a HEAD@{5}: commit: before running on sliding windows 2
# 2c092cf HEAD@{6}: commit: before running on sliding windows 1
# 425f674 HEAD@{7}: commit: before running on sliding windows
# de5302d HEAD@{8}: commit: 'commit'

git reset --hard CommitTag

# CommitTag is the leftmost coulumn
```



## Building a .gitignore

In case one wants to add files which are bigger that say 2Mb to .gitignore, a nice command is:

```bash
find . -size +2M | sed 's|^\./||g' | cat >> .gitignore
```

#### Manually adding files to the .gitignore:

An example of a few lines one will want to add to a .gitignore file:

```bash
*.mex*
*.o
*.a
*.cai
*.exe
*.dll
*.pdf
*.obj
*.so
*.lib
*.rar
*.zip
*.tar
*.tgz
*.7z
*.7zipreer
*.sbr
*.exp
*.ilk
*.pdb
*.res
*.tlog
*.log
*.idb
*.cache
*.enc
*.lastbuildstate
*.pyc
*.npy
*.doc
*.xls
*.docx
*.xlsx
*.ppt
*.pptx
*.vsd
*.hex
*.dng
*.raw
*.dvs
*.avi
*.mov
*.mp4
*.mp3
*.mp2
*.mpg
*.mpeg
*.mat
*.idea
*.pth
```

