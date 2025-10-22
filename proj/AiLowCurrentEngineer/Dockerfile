# ===== Stage 1: build with Maven =====
FROM maven:3.9.6-eclipse-temurin-21 AS build
WORKDIR /app

# Кэшируем зависимости
COPY pom.xml .
RUN mvn -q -B -DskipTests dependency:go-offline

# Сборка
COPY src ./src
RUN mvn -q -B -DskipTests package

FROM eclipse-temurin:21-jre
WORKDIR /app
# Копируем собранный .jar из предыдущего шага
COPY --from=build /app/target/*.jar /app/app.jar
EXPOSE 8080
ENTRYPOINT ["java","-jar","/app/app.jar"]
