# Docker Environment

This folder contains a dockerfile that installs a Python 3.6, Keras 2.0.5 and TensorFlow 1.2.1 environment, so that the scripts in the 2017-07 seminar folder can be executed.


## Cheatsheet

Build the docker container:
```
docker build ./docker/dockerfile -t dl4nlp
```


Run the container and mount the current folder ${PWD} into the container:
```
docker run -it -v ${PWD}:/usr/src/app dl4nlp bash
```

