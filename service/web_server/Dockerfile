FROM openjdk:8-jdk-alpine

WORKDIR /ybigta/program/bootapp
ADD build/libs/alpacapaca-app.jar app.jar

EXPOSE 10001
VOLUME ["/ybigta/logs/bootapp", "/ybigta/logs/bootapp/gc"]

ENV JAVA_OPTS=""
CMD ["/bin/sh", "-c", "exec java $JAVA_OPTS -jar app.jar >> /ybigta/logs/bootapp/app.log 2>&1"]