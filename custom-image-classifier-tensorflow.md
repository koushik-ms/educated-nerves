# Custom Image Classifier Using Tensorflow: Distinguishing Superheroes of one generation from the next

This is a summary of the experiences & learnings from my experiment a few months ago building a tensorflow-based image classifier. 

## TL; DR 
Jump [here](#steps-to-create-the-custom-image-classifier-built-for-this-experiment) to learn how you can build your own in 1, 2,...5!

## Ready to Shoot!

Everytime I see a smartphone camera draw a neat rectangle around my face, it reminds me of the day I graduated from DOOM to the FPS of 90s that would pin-point the enemy with a neat rectangle around him. In either case, a neat rectangle means I'm ready to shoot!

I have always been fascinated by object detection and face-recognition algorithms and having experimented since early days using [OpenCV/Haar Cascade classifiers](https://docs.opencv.org/3.1.0/d7/d8b/tutorial_py_face_detection.html) and the [Viola-Jones](https://www.youtube.com/watch?v=_QZLbR67fUU) face-detection algorithms, it’s amazing how far the algorithms have come in the last few years. [Convolutional Neural Networks](https://pjreddie.com/darknet/yolo/) and [supervised learning approaches](https://github.com/tensorflow/models/tree/master/research/object_detection) have really improved the training and detection performance multifold in several areas (speed, accuracy, space & complexity to name a few) and it’s not unfair to say that today, given my last 10 years photo collection, the facebook’s face ID algorithm will do far better than me in recognizing my old friends :).

When it comes to machine learning frameworks, I am particularly drawn to [tensorflow by Google](https://www.tensorflow.org/) and was thus delighted when I came across [this codelab](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#0) & [tutorial](https://www.tensorflow.org/tutorials/keras/basic_classification) promising to teach anyone the basics of building their own image classifier. This really dense [video](https://www.youtube.com/watch?v=QfNvhPx5Px8) pushed me over any remaining hesitation and I set about training my own CNN for a custom Image classifier. 

## Distinguishing superheros

Can you tell the difference between [Darth Vader](https://en.wikipedia.org/wiki/Darth_Vader) and [Elsa](https://en.wikipedia.org/wiki/Elsa_(Frozen)) ? Of course, even a 3rd grader like my daughter could do that. In fact it was in one of those discussions with her about Elsa and her pervasiveness in my daughter’s life then, that the conversation veered to Star Wars (my childhood fascination) and the mighty Darth Vader. Given my daughter couldn’t understand, for the life of her, why Darth Vader would be any cool, we were finally even... and I had an idea!

What if I train a classifier to distinguish between Darth Vader and Elsa using Tensorflow ? Thus, this experiment began.

## Start your engines

To setup a training environment for a classifer we need a recent Tensorflow version and lots of training data. The system requirements are quite frugal and an old desktop with an i3 and 8GB of RAM would do just fine. All the hard-work for putting a training environment together has been done by [xblaster](https://github.com/xblaster) and his [docker image](https://hub.docker.com/r/xblaster/tensor-guess/) is all we need to get started.

```bash
$ docker pull xblaster/tensor-guess
```

This will download a few GB of data from the internet to get the different layers of this docker image. This image is built on top of the official tensorflow image from the tensorflow authors and adds some convenient helper scripts to make training and testing easy. You can read more about [docker](https://docs.docker.com/engine/docker-overview/) or this docker image [here](https://github.com/xblaster/tensor-guess).

## Data is the new Oil!

To train an accurate classifier requires a lot of training data of good quality. In my case I was looking for images with Darth Vader and Elsa. And, yes, Google is my best friend. I made sure I included some non-standard images e.g., [this][1] and [this][2] for Darth Vader and [this][3] and [this][4] for Elsa. For each category, I downloaded about 250-300 images in relatively small sizes (300x150 or smaller) and set aside about 10% of this for testing and used the remaining images for training. More than the quantity, it’s really important to have quality training & test images – links to resources for training /test data.

    [1]: https://github.com/koushik-ms/tensorflow_image_classifier/raw/master/tf_files/data/darth_vader/s1images(70).jpg
    [2]: https://github.com/koushik-ms/tensorflow_image_classifier/raw/master/tf_files/data/darth_vader/s1images2d323d2d3c23d.jpg
    [3]: https://github.com/koushik-ms/tensorflow_image_classifier/raw/master/tf_files/data/elsa/images33.jpg
    [4]: https://github.com/koushik-ms/tensorflow_image_classifier/raw/master/tf_files/data/elsa/images(34).jpg

Once you have collected the training data, put the images in a training folder organized into one sub-folder per category (where category name is the sub-folder name). For this experiment the training folder is named `tf_files`.

```
- tensorflow_image_classifier
    + src
    - tf_files
        - data
            + darth_vader
            + elsa
```

If you’d like to use the training & test data I used then just clone [this github repository](https://github.com/koushik-ms/tensorflow_image_classifier).

```bash
$ git clone https://github.com/koushik-ms/tensorflow_image_classifier.git
$ cd tensorflow_image_classifier
```
## Training the image classifier

We will be using supervised learning: this means we will initially supply a set of images that are labelled with the categories they represent to help the CNN learn more about the categories. Also, we will use the transfer learning approach as explained in [this codelab](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#0) - we will use the inception model that has been trained to classify images (of flowers) and retrain it to classify images of our superheroes. This approach drastically reduces the training time for deep multilayered neural networks. This also shows how CNNs generalize well across classification problems by their ability to progressively code more and more intricate features into deeper layers of the network.

Tensorflow comes with an example python script that we can leverage for this purpose. To start this training on the cloned repository, just run

```bash
$ bash ./train.sh $PWD/tf_files
```

We need to provide absolute paths in parameters to train.sh and other scripts since they internally use them to mount these directories as volumes for the docker container and docker requires absolute path names.

This will start the training process which runs for 4000 iterations. The script will initially download the inception model, then create bottleneck files (which represents the newly trained layer to adapt this generic classifier for our specific problem) and then run the training iterations. The script outputs the training results at the end of each iteration. On a 3-yr old desktop machine with an i3-4150 3.5GHz processor it took close to an hour to finish the training process whereas on a fairly recent 8th gen i7-8550U 4GHz processor, it took about 11 minutes. Either case, it is a fraction of the time typically taken to train such complex models and this gain is entirely due to the transfer learning approach.

## Testing the classifier

With training completed, it’s time to see how our classifier performs. To classify an image run

```bash
$ bash ./guess.sh $PWD/tf_files $PWD/test_data/images00.jpg
```

where images00.jpg is the file we want to test:
[image]

The script outputs:
<< output >>

which is pretty accurate.

To guess an entire set of images, we can run it on a directory full of images giving it another location to save classified images. Each test image is copied to this location and renamed to <category>-[score].<ext> where <category> is the predominant category into which the image was classifed (one of “darth_vader” or “elsa”) and [score] is a number representing how highly it fits into the category. For example,

```bash
$ mkdir classified
$ bash ./guessDir.sh $PWD/tf_files $PWD/test_data $PWD/classified
```

Note: There are scripts with name train.sh, guess.sh etc at the top-level directory of the repo and inside the src folder. Invoke the ones in the top-level dir – these provide a wrapper around the scripts in src directory.

That’s it! The `classified` folder now contains labelled images as classified by our custom classifier. 

Below is a quick re-cap of all the steps.

## Steps to create the custom image classifier built for this experiment

1. Get the docker image `docker pull xblaster/tensor-guess`
1. Clone the repo: `git clone https://github.com/koushik-ms/tensorflow_image_classifier.git && cd tensorflow_image_classifier`
2. Train the classifier: `bash ./train.sh $PWD/tf_files`
3. Test with 1 image: `bash ./guess.sh $PWD/tf_files $PWD/test_data/images00.jpg`
4. Test with entire test data set: `mkdir -p classifier && bash ./guessDir.sh $PWD/tf_files $PWD/test_data $PWD/classified`

## Analysing test results

While looking at the classifier performance on test images, it performs surpisingly well even on images that have many deviations, for example: 
![this](https://github.com/koushik-ms/tensorflow_image_classifier/raw/master/test_data/s2images87.jpg), ![this](https://github.com/koushik-ms/tensorflow_image_classifier/raw/master/test_data/images90.jpg) and ![this](https://github.com/koushik-ms/tensorflow_image_classifier/raw/master/test_data/imagesws.jpg). 

One particular test image throws a real challenge:

![Challenge](https://github.com/koushik-ms/tensorflow_image_classifier/raw/master/test_data/imagesfg.jpg "Challenging Test Image")

This image contains both the features of Darth Vader (sharp lines, dark colors, the bottom half of a pentagon) and those of Elsa (dress color, pixie dust, ...). Fittingly, the classifier assigns a hybrid score to it, although it assigns a higher score for Darth. 

$ bash guess.sh $PWD/tf_files $PWD/test_data/imagesfg.jpg
<<output>>

A human observer will have no trouble associating this picture with Elsa (given the categories) but the classifier is misled. The cause for this is clear – other than the presence of the features explained above the prominent eyes of Elsa are missing. A quick hack confirms this: creating another test-image correcting some of these errors (white background) and adding placeholder eyes (just black ellipses above the neck area) and giving it to our classifier makes it “see” Elsa in the image:

$ bash guess.sh $PWD/tf_files $PWD/test_data/imagesff.jpg
<<output>>

This is a bias in our training data set in which there were very few pictures of Elsa’s dress alone and none of her from the back without her face showing in the picture.

This provides us a clue of how the training data can be improved to improve the accuracy of classifier. 

## On Superheroes of the time

Every generation has its own superheroes which reflect the collective psyche of the generation replete with values it holds dear, the environment it operates in and the challenges it must overcome. Every generation has its own challenges and cherishes its own superheroes that get the job done.

Pictures say a thousand words and whether in art, photography or education, images play a key role. Every generation, thus, has been faced with and has overcome a significant challenge in image processing. In the 90s it was image compression and encoding breakthroughs that led to the ubiquitous digital multimedia culture and in the past decade it has been image classification and object detection. If codecs were the superheroes of yesteryears then the current generation belongs to machine learning frameworks like tensorflow and to neural networks.

Hope this tutorial was useful to you and helps you gets closer to your goal in understanding and applying machine learning. I would like to hear your comments and see where you will apply this – go ahead and fork the github repository and make it your own. Link to your model (tf_files/retrained_graph.pb) in github or share it.


Useful links:
https://medium.com/@nikasa1889/the-modern-history-of-object-recognition-infographic-aea18517c318
https://github.com/Nikasa1889/HistoryObjectRecognition
https://github.com/powerline/fonts
