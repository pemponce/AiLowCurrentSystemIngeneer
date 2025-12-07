package com.example.ailowcurrentengineer.config;

import com.example.ailowcurrentengineer.model.Role;
import com.example.ailowcurrentengineer.repository.RoleRepository;
import org.springframework.boot.CommandLineRunner;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class AuthInit {
  @Bean
  CommandLineRunner initRoles(RoleRepository roles) {
    return args -> roles.findByName("ROLE_USER").orElseGet(() -> roles.save(Role.builder()
        .name("ROLE_USER")
        .build()));
  }
}
