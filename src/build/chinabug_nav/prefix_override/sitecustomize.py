import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/just-vitou/turtlebot3_workspace/src/install/chinabug_nav'
