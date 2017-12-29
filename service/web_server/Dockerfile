FROM openjdk:8-jdk-alpine

WORKDIR /ybigta/program/bootapp
ADD set-java-env.sh set-java-env.sh
ADD build/libs/alpacapaca-app.jar app.jar

EXPOSE 11302
VOLUME ["/ybigta/logs/bootapp", "/ybigta/logs/bootapp/gc"]

CMD ["/bin/sh", "-c", "source /ybigta/program/bootapp/set-java-env.sh && exec java $JAVA_OPTS -jar /ybigta/program/bootapp/app.jar >> /ybigta/logs/bootapp/app.log 2>&1"]