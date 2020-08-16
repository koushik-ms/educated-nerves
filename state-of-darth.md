# State of Darth

A little over two years ago as I was getting to know Machine learning, I had a chance to train a CNN to classify superheroes. At that time, I used the "inception" model which was the state of the art and trained it using the transfer learning approach. The whole thing took me 3-4 days to finish including training the model (I am not sure if I had a GPU then) and writing the post.

Recently I had a chance to repeat this experiment. As I was going through the fast.ai course titled "Practical Deep Learning for developers", a lesson presented a really simple approach to transfer learning using the resnet34 (arguably, the current state-of-the-art for image classification) and I decided to try it on the dataset from 2 years ago.

In less than 20 minutes I was able to finish training the model on the cloud. The setup, training/test separation, data augmentation etc was really simple. 

There was only one problem - even after having the number of images or depth of the NN (resnet34 -> 18) the model accuracy was close to 100%. So I decided to throw batman into the mix to level the problem up a little bit. Nevertheless, the model trained quite quickly and was able to classify with ~95% accuracy. Even for the images that were too tricky for the older model the new one had no problems. 

conclude about state of art

The key takeaway for me is how the state-of-the-art makes it almost trivial to include learning/ inference as part of any software components. For enthusiasts like me who view DL as a tool (rather than the core domain) this makes me wonder why not make learning and adapting a feature of every piece of software where it makes sense.

recommend fastai course

Finally I can't recommend the fast.ai course enough for anyone interested in applying DL into their field / area of work. Regardless of level of math and previous ML/AI exposure the course gets you up and running with practical tools that can be used from day one. If you will do one ML/DL course, let it be this one 



Ps. I'm in no way associated with fast.ai just a happy student.


