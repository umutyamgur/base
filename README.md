# base
base setup for gpu based machine learning experiments with docker for remote dgx machines

# Setup Instructions
Useful scripts needed below are in "scripts" folder.
1. Fork this repository and pull to the dgx machine.
2. Install pip from get-pip.py, ensure pip is on your path
3. Install gpustat and pdm
4. Install docker compose with install_docker_compose.sh
5. Copy "cpu" and "gpu" scripts to home/bin and ensure they are executable and on your path
6. run "docker compose up -d" to build and start the docker container
7. run "cpu pdm exec examples/show_devices.py" to verify that only cpu is visible. should print "cuda is not available"
8. run "gpu pdm exec examples/show_devices.py" to verify that exactly one gpu is visible. should print "cuda is available, with 1 devices, current device index 0 (Tesla V100-SXM3-32GB-H)"
9. put library files in "base/...", e.g. pytorch module definitions in "base/models.py"
10. put experiments which use these files in "examples/..."
11. ...
12. profit.
