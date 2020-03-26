
# Some clues for using git together, as a team.

"Team, team, team." (Some episode of IT crowd.)

First you have to create an account for you in github.
Then go to the patternrec_2020-samh du github de Hanna 
(link she gave in the forum on ilias).

And also install git on your computer.

Then, in your terminal, from where you want to upload the repository: 
```bash
git clone git@github.com:hannaglad/patternrec_2020-samh.git 
# the last thing is the link of the repository
# you can find it when pressing the "clone or download" green button
# for my part (saskia), I am using a ssh key. 
``` 

### the ssh key

I am not sure, but I vaguely remember that it becomes a little bit 
easier if you add to your github account a ssh key somewhere. 
I don't remember by heart on how you can do it, but it must be some 
tutorials somewhere in the internet.

## Forking things

Then go back to the github and click fork in order to have your own fork 
of the project.

Then, in the terminal, 
```bash
git remote add name_of_your_fork link_of_your_fork
# I (saskia) suggesting that we use our name as the name_of_our_fork.
# the link_of_your_fork is find 
# also by pressing the famous green button of your fork
``` 

Then, always in your terminal, you can change things like modifying an 
already existing file or making a new one. 

Tips 1: Sometimes, it is easier to just use vim for small changes.

Then use:
```bash
git add file.txt # to update what will be commited
# you can add multiple modified file also or not
# depending on what you want to be part of your next commit
# and for commiting it:
git commit -m "some comments on what you have modified"
```  

### ```git status``` 

At anytime, it is recommended to use ```git status``` to have some infos 
like what could be added, what could be commited,...

For uploading the commited files back to github:
```bash
git push name_of_your_fork
``` 

Then, go to the patternrec_2020-samh de Hanna again. 
Go to the onglet pull requests, press "new pull request" green button. 
Here you can ask for a merging of your fork with the one of Hanna.

Then, Hanna, our great leader with all her wisdom ;) , 
has to accept the merging.

Don't be afraid to modified this document just to practice and/or also 
to add some new tips or a different way of how we can use git together.



