FROM python:3.6-slim

# ensure local python is preferred over distribution python
ENV PATH /usr/local/bin:$PATH
ENV PYTHON_VERSION 3.6

# install runtime dependencies, make some useful symlinks that are expected to exist, update pip, setuptools, and wheel
RUN set -eux && \
    apt-get update && \
    apt-get install --no-install-recommends -y \
    make \
    wget \
    gcc \
    vim && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
#    cd /usr/local/bin && \
#    ln -s python3 python && \
#    ln -s python3-config python-config && \
    pip install -U \
    pip \
    setuptools \
    wheel

# set the working directory in the container
WORKDIR /src

# create a new user
RUN useradd -m -r user && \
    chown user /src
USER user

# copy the dependencies file to the working directory
COPY requirements.txt .

# install dependencies
RUN pip install -r requirements.txt

# copy the content of the src directory to the working directory
COPY src/ .
COPY app.py .
COPY test_app.py .
COPY dataset/ dataset/
COPY trained_models/ trained_models/

# Open a bash terminal on running the docker image
CMD ["/bin/bash"]

EXPOSE 5000

### Some helpful docker commands ###
## To build the image
# time docker build -t mail_classifier -f Dockerfile --quiet .

# docker run --rm -p 8080:8080 mail_classifier /bin/bash

## See build images
# docker images

## See running processes
# docker ps

## Delete an image
# docker rmi <image_name>

## Stop a running container
# docker stop <container id>

## Delete a container
# docker rm <container id>

## TODO: Try multi-stage builds to reduce size of docker images

# Entering into a bash shell
# docker run --rm -it --entrypoint bash mail_classifier
