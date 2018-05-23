import os


def setup_conda_env():
    path_to_environment_file = os.path.normpath(
            os.path.join(os.path.abspath(__file__), '../../../environment.yml'))
    cmd = 'conda env create -f %s' % path_to_environment_file
    os.system(cmd)

