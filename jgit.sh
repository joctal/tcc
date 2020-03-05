#!/bin/bash
git config --global core.preloadindex true
git config --global core.fscache true
git config --global gc.auto 256
	git add *
if [ `git status | grep -e "modified" -e "new file" | wc -l` -gt 0 ];then
	git pull
	git status
	git add *
	git add .gitignore
	git add robot_copia_codigos_FULL.sh

	#git commit -a
	git commit -m "jss_servidor_tangram $1"
	git push -u origin master
	git status
fi


