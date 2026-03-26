FROM ubuntu:latest
LABEL authors="darko"

ENTRYPOINT ["top", "-b"]