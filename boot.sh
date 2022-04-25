#! /bin/sh
sudo apt install python-pip
sudo apt install python3-dev
sudo apt install python3.10-venv


python3 -m venv env
source env/bin/activate


pip install -r requirements.txt
