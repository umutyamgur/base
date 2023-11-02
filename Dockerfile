# select image, we choose tensorflow image to build off of for JAX
FROM nvcr.io/nvidia/pytorch:23.10-py3 AS original
FROM original as builder

# install python installation tools
RUN pip install --upgrade pip setuptools wheel
RUN pip install pdm

# copy in files defining the build for pdm
COPY pyproject.toml pdm.lock README.md /base/
# COPY senn/ /senn/senn

# set working directory and install all dependencies with pdm
# note that local project package is *not* installed
WORKDIR /base
RUN mkdir __pypackages__ && pdm sync --no-editable --no-self


# select image, same as above
FROM original

# set python path to new directory and copy pdm installation from builder into it
ENV PYTHONPATH=/pdm_pkgs
COPY --from=builder /base/__pypackages__/3.10/lib /pdm_pkgs

# set working directory
WORKDIR /base

# entrypoint is bash so that we get an interactive shell by default
# /root/.bashrc can be used to run shell commands on startup
ENTRYPOINT "/bin/bash"
