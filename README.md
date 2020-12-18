# MiniStack

## Task-1
Search Engine on Stackoverflow corpus nearly 160k+ docs

I split the code in three notebook files
- DataExtraction is code for collecting data
- DataPreprocessing is code for processing the data
- Retrieval is code for retrieving top 10 similar docs

## Required libraries and packages are
 pandas, numpy, sklearn, nltk, re, os ,sys, csv, xml

## Dataset
For this project I collected data from  [Stack Exchange Data Dump website] (https://archive.org/download/stackexchange)

## Task-2
A web crawler which crawls the stackoverflow website(https://stackoverflow.com/questions?page=1) and finds the most popular technologies at current point of time by getting the tags information of the newest questions asked on the website.

webcrawler is the code for the this task

## Required libraries are
urllib3, requests, bs4, zlib, operator, os, sys

## How to run
Download the files and make sure all the files and folders are in the same directory

## UI Demo
To run the code in server
- Go to UI-demo folder
- create a virual environment  ( Command : virtualenv env for windows)
- activate the virtual environment (Command : env/Scripts/activate)
- install requirements.txt (Command: pip install -r requirements.txt)
- run python app.py in the terminal

## Github repository Link
https://github.com/Saideepthi123/MiniStack
