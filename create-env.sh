#!/bin/bash

# Create a virtual environment in parent directory
current_dir=`pwd`
parent_dir="$(dirname "$current_dir")"

# Create environment
venv_dir=${parent_dir}/venv-breakingws-py3
/usr/bin/python3 -m venv --upgrade --without-pip ${venv_dir} 

# Activate the environment
echo $venv_dir
source ${venv_dir}/bin/activate

# Install pip
curl https://bootstrap.pypa.io/get-pip.py | python

# Check python version and pip list
deactivate
source ${venv_dir}/bin/activate
pip3 install --upgrade pip

# Intall requirements
pip3 install -r ${current_dir}/requirements.txt
deactivate

echo
echo "Virtual environment 'venv-breakingws-py3' created!"
echo 
echo "To setup the package type 'source setup.sh'."
echo

