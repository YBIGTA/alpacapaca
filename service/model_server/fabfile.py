# fabfile.py
from fabric.contrib.files import append, exists, sed, put
from fabric.api import env, local, run, sudo
import os
import json

# 현재 fabfile.py가 있는 폴더의 경로
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# deploy.json이라는 파일을 열어 아래의 변수들에 담아줍니다.
envs = json.load(open(os.path.join(PROJECT_DIR, "deploy.json")))

REPO_URL = envs['REPO_URL']
PROJECT_NAME = envs['PROJECT_NAME']
REMOTE_HOST = envs['REMOTE_HOST']
REMOTE_HOST_SSH = envs['REMOTE_HOST_SSH']
REMOTE_USER = envs['REMOTE_USER']
KEY_FILENAME = envs['KEY_FILENAME']

# SSH에 접속할 유저를 지정하고,
env.user = REMOTE_USER
# SSH로 접속할 서버주소를 넣어주고,
env.hosts = [
    REMOTE_HOST_SSH,
]
env.key_filename = KEY_FILENAME
# 원격 서버중 어디에 프로젝트를 저장할지 지정해준 뒤,
project_folder = '/home/{}/{}'.format(env.user, PROJECT_NAME)
server_folder = '/home/{}/{}/service/model_server'.format(env.user, PROJECT_NAME)
virtualenv_dir = '/home/{}/.virtualenvs/{}'.format(env.user, PROJECT_NAME)
# 우리 프로젝트에 필요한 apt 패키지들을 적어줍니다.
apt_requirements = [
    'curl',
    'git',
    'python3-dev',
    'python3-pip',
    'build-essential',
    'nginx'
]

# _로 시작하지 않는 함수들은 fab new_server 처럼 명령줄에서 바로 실행이 가능합니다.
def new_server():
    setup()
    deploy()

def setup():
    _get_latest_apt()
    _install_apt_requirements(apt_requirements)
    _make_virtualenv()

def deploy():
    _get_latest_source()
    _put_envs()
    # _download_data()
    _update_virtualenv()
    _enable_site()
    _emperor_mode()
    _startup_boot()
    _grant_nginx()
    _restart_nginx()

# apt 패키지를 업데이트 할 지 결정합니다.
def _get_latest_apt():
    update_or_not = input('would you update?: [y/n]')
    if update_or_not == 'y':
        sudo('apt-get update && apt-get -y upgrade')

# 필요한 apt 패키지를 설치합니다.
def _install_apt_requirements(apt_requirements):
    reqs = ' '.join(apt_requirements)
    sudo('apt-get -y install {}'.format(reqs))
    sudo('/etc/init.d/nginx start')

# virtualenv와 virtualenvwrapper를 받아 설정합니다.
def _make_virtualenv():
    if not exists('~/.virtualenvs'):
        script = '''"# python virtualenv settings
                    export WORKON_HOME=~/.virtualenvs
                    export VIRTUALENVWRAPPER_PYTHON="$(command \which python3)"  # location of python3
                    source /usr/local/bin/virtualenvwrapper.sh"'''
        run('mkdir ~/.virtualenvs')
        sudo('pip3 install virtualenv virtualenvwrapper')
        run('echo {} >> ~/.bashrc'.format(script))

# Git Repo에서 최신 소스를 받아옵니다.
# 깃이 있다면 fetch를, 없다면 clone을 진행합니다.
def _get_latest_source():
    if exists(project_folder + '/.git'):
        run('cd %s && git fetch' % (project_folder,))
    else:
        run('git clone %s %s' % (REPO_URL, project_folder))
    current_commit = local("git log -n 1 --format=%H", capture=True)
    run('cd %s && git reset --hard %s' % (project_folder, current_commit))

# put이라는 방식으로 로컬의 파일을 원격지로 업로드할 수 있습니다.
def _put_envs():
    put('uwsgi_params', os.path.join(server_folder, 'uwsgi_params'))
    put('mywebsite_nginx.conf', os.path.join(server_folder, 'mywebsite_nginx.conf'))
    put('mywebsite_uwsgi.ini', os.path.join(server_folder, 'mywebsite_uwsgi.ini'))

def _download_data():
    pass
# run('scp -i ~/key/ec2-dreamgonfly.pem ~/Documents/ybigta/alpacapaca/service/model_server/requirements.txt ubuntu@52.38.217.70:~/alpacapaca/service/model_server/.')
    # run('scp -i ~/key/ec2-dreamgonfly.pem ~/Documents/ybigta/alpacapaca/train/output/reader_params.pkl ubuntu@52.38.217.70:~/alpacapaca/train/output/.')
    # run('scp -i ~/key/ec2-dreamgonfly.pem ~/Documents/ybigta/alpacapaca/train/output/saved_model.pkl ubuntu@52.38.217.70:~/alpacapaca/train/output/.')

# Repo에서 받아온 requirements.txt를 통해 pip 패키지를 virtualenv에 설치해줍니다.
def _update_virtualenv():
    if not exists(virtualenv_dir + '/bin/pip'):
        run('cd /home/%s/.virtualenvs && virtualenv --system-site-packages -p python3 %s' % (env.user, PROJECT_NAME))
    sudo('%s/bin/pip install -r %s/requirements.txt' % (    
        virtualenv_dir, server_folder
    ))

def _enable_site():
    if not exists('/etc/nginx/sites-enabled/mywebsite_nginx.conf'):
        sudo('ln -s {} /etc/nginx/sites-enabled/'.format(os.path.join(server_folder, 'mywebsite_nginx.conf')))

def _emperor_mode():
    if not exists('/etc/uwsgi'):
        sudo('mkdir /etc/uwsgi')
    if not exists('/etc/uwsgi/vassals'):
        sudo('mkdir /etc/uwsgi/vassals')
    if not exists('/etc/uwsgi/vassals/mywebsite_uwsgi.ini'):
        sudo('ln -s {} /etc/uwsgi/vassals/'.format(os.path.joinn(server_folder, 'mywebsite_uwsgi.ini')))

script = """'#!/bin/sh -e
#
# rc.local
#
# This script is executed at the end of each multiuser runlevel.
# Make sure that the script will "exit 0" on success or any other
# value on error.
#
# In order to enable or disable this script just change the execution
# bits.
#
# By default this script does nothing.
/home/ubuntu/.virtualenvs/alpacapaca/bin/uwsgi --emperor /etc/uwsgi/vassals --uid www-data --gid www-data --daemonize /var/log/uwsgi-emperor.log
exit 0'"""

def _startup_boot():
    sudo('echo {} > /etc/rc.local'.format(script))
    # first start
    sudo('/home/ubuntu/.virtualenvs/alpacapaca/bin/uwsgi --emperor /etc/uwsgi/vassals --uid www-data --gid www-data --daemonize /var/log/uwsgi-emperor.log')
    
# NginX가 프로젝트 파일을 읽을 수 있도록 권한을 부여합니다.
def _grant_nginx():
    sudo('chown -R :www-data ~/{}'.format(PROJECT_NAME))
    sudo('chmod -R 775 ~/{}'.format(PROJECT_NAME))

# 마지막으로 NginX를 재시작합니다.
def _restart_nginx():
    sudo('/etc/init.d/nginx restart')