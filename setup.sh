
#!/bin/bash

# See this stackoverflow question
# http://stackoverflow.com/questions/59895/getting-the-source-directory-of-a-bash-script-from-within
SETUP_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

#
# Base package root. All the other releavant folders are relative to this
# location.
#
export BREAKINGWS_ROOT=$SETUP_DIR
echo "BREAKINGWS_ROOT set to " $BREAKINGWS_ROOT

#
# Add the root folder to the $PYTHONPATH so that we can effectively import
# the relevant modules.
#
export PYTHONPATH=$BREAKINGWS_ROOT:$PYTHONPATH
echo "PYTHONPATH set to " $PYTHONPATH
