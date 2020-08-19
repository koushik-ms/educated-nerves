# State of Darth

When it comes to Deep Learning, does the state-of-the-art make it any easier to adopt DL today than it was, say, two years ago ? Should you even be bothered about starting to learn ML now, if you haven't yet had a chance to explore and use it in your work/ personal projects ? 

The answers to both these questions is, as I realized recently, a resounding yes and the state-of-the art is about a 100x better than what it was 2 years ago. How do I know that ? Well, A few weeks ago I had a chance to repeat an exercise I did while getting introduced to ML - a [transfer learning approach](https://github.com/koushik-ms/educated-nerves/blob/master/custom-image-classifier-tensorflow.md) to train a deep learning model to classify images of "Darth Vader" and "Elsa". As I was going through the fast.ai [course](https://course.fast.ai/) titled "Practical Deep Learning for developers", I came across a [simple approach to transfer learning](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson2-download.ipynb) using the resnet34 (arguably, the current state-of-the-art for image classification) and I decided to try it on the same problem.

What had taken about 4 days from start to finish in 2018 was over in all of 30 mins, including the setup, data augmentation, even training the model on a GPU in the cloud. 

Except, there was one "problem" -  the accuracy was too high. It was close 99% even with half the images and half the number of hidden layers (resnet18) so I added batman to the mix to level the problem up a bit. One more iteration and 20 mins later (with just auto-downloaded Google image search results) the model was accurately classifying the three classes - even the images that were too tricky for the earlier model. The final notebook is available [here](https://github.com/koushik-ms/educated-nerves/blob/master/Superheroes.ipynb).

The key takeaway for me was the 100x reduction in the time it took and the ease of building something like this thanks to constant research and blossoming of new tools & services (shout out to [Gradient](https://gradient.paperspace.com/), [colab](http://colab.research.google.com/) and [fastai](https://docs.fast.ai/)). Thus, It is constantly getting easier to plug learning and inference as a feature into any area of software that could benefit from it (even something as ubiquitous as, say, [code auto-completion](https://www.kite.com/)).

Finally, I'd strongly recommend the fast.ai course for anyone interested in learning how to apply deep learning into their own area of work. If you're willing to give only one DL course a try, let it be this one. 

Ps. Many thanks to Himanshu and Sameer for introducing me to this.