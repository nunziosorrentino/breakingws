current_dir=`pwd`
parent_dir="$(dirname "$current_dir")"
venv_dir=${parent_dir}/venv-breakingws-py3

# First activate the virtual enviroment, if created

source ${venv_dir}/bin/activate

# Base package root. All the other releavant folders are relative to this
# location.

export BREAKINGWS_ROOT=$current_dir
echo "BREAKINGWS_ROOT set to " $BREAKINGWS_ROOT
echo
# Add the root folder to the $PYTHONPATH so that we can effectively import
# the relevant modules.
export PYTHONPATH=$BREAKINGWS_ROOT:$PYTHONPATH
echo "PYTHONPATH set to " $PYTHONPATH
echo
# Add the bin folder to the $PATH so that we have the executables off hand.
#
export PATH=$BREAKINGWS_ROOT/breakingws/bin:$PATH
echo "PATH set to " $PATH
echo
echo "BreakinGWs setup done!"
echo
echo "Type 'deactivate' to exit the environment!"
echo

