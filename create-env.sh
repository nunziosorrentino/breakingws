#!/bin/bash

# Create a virtual environment in parent directory
current_dir=`pwd`
parent_dir="$(dirname "$current_dir")"

venv_dir=${parent_dir}/venv-breakingws-py3
/usr/bin/python3 -m venv ${venv_dir}
echo $venv_dir
source ${venv_dir}/bin/activate

pip3 install --upgrade pip
pip3 install -r ${current_dir}/requirements.txt
deactivate

echo "Virtual environment 'venv-breakingws-py3' created! To setup the package:"
echo "source setup.sh"

