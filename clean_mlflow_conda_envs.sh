#!/bin/bash
Help()
{
    echo "Syntax: scriptTemplate [-d|l|h]"
    echo
    echo "-d: delete conda envs starting with 'mlflow'."
    echo "-l: list conda envs starting with 'mlflow'."
    echo "-h: this help message."
    echo 
}

while getopts l:d:h: flag

do 
    case "${flag}" in
        h)  # show help
            Help
            exit;;

        l)  # list relevant conda environments
            for i in $( conda env list | grep -e "mlflow-" );  do echo $i; done
            exit;;

        d)  # delete relevant conda environments
            for i in $( conda env list | grep -e "mlflow-" ); do conda remove -n $i --all; done
            exit;;

        \?) # Invalid option
            echo "Error: Invalid option"
            exit;;
    esac
done