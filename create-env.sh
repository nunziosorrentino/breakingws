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

# Create a setup script in current directory
echo "source ${venv_dir}/bin/activate" > ${current_dir}/setup.sh
#echo "export PATH=${current_dir}/bin:\$PATH" >> ${current_dir}/setup.sh
echo "export PYTHONPATH=${current_dir}:\$PYTHONPATH" >> ${current_dir}/setup.sh
echo "echo">> ${current_dir}/setup.sh
echo "echo \"breakingws setup done!\"">> ${current_dir}/setup.sh
echo "echo">> ${current_dir}/setup.sh
echo "echo \"Type 'deactivate' to exit the environment!\"">> ${current_dir}/setup.sh
echo "echo">> ${current_dir}/setup.sh


echo "Virtual environment 'venv-breakingw-py3' created! To setup the package:"
echo "source setup.sh"

