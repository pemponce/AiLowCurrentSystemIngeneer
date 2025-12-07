package com.example.ailowcurrentengineer.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.reactive.function.client.WebClient;

@Configuration
public class PlannerClientConfig {
  @Bean
  public WebClient plannerWebClient(@Value("${planner.url}") String baseUrl) {
    return WebClient.builder().baseUrl(baseUrl).build();
  }
}
