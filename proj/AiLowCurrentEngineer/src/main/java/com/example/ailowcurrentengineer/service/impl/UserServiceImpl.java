// com/example/ailowcurrentengineer/service/impl/UserServiceImpl.java
package com.example.ailowcurrentengineer.service.impl;

import com.example.ailowcurrentengineer.Auth.Jwt.JwtService;
import com.example.ailowcurrentengineer.model.Role;
import com.example.ailowcurrentengineer.model.User;
import com.example.ailowcurrentengineer.repository.RoleRepository;
import com.example.ailowcurrentengineer.repository.UserRepository;
import com.example.ailowcurrentengineer.service.UserService;
import lombok.RequiredArgsConstructor;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;

import java.util.Map;

@Service
@RequiredArgsConstructor
public class UserServiceImpl implements UserService {

    private final UserRepository userRepository;
    private final RoleRepository roleRepository;
    private final PasswordEncoder passwordEncoder;
    private final JwtService jwtService;

    @Override
    public void createUser(String fullName, String email, String password) {
        if (!StringUtils.hasText(email) || !StringUtils.hasText(password)) {
            throw new IllegalArgumentException("Email and password are required");
        }
        if (userRepository.existsByEmail(email)) {
            throw new IllegalArgumentException("User already exists");
        }
        Role userRole = roleRepository.findByName("ROLE_USER")
                .orElseThrow(() -> new IllegalStateException("ROLE_USER not found, проверь миграцию/инициализацию"));

        User u = User.builder()
                .email(email)
                .fullName(fullName)
                .password(passwordEncoder.encode(password)) // <-- хэшируем
                .enabled(true)
                .build();
        u.getRoles().add(userRole);
        userRepository.save(u);
    }

    @Override
    public String login(String email, String password) {
        if (!StringUtils.hasText(email) || !StringUtils.hasText(password)) {
            throw new IllegalArgumentException("Email and password are required");
        }
        User u = userRepository.findByEmail(email)
                .orElseThrow(() -> new IllegalArgumentException("Invalid credentials"));

        if (!u.isEnabled()) {
            throw new IllegalStateException("User is disabled");
        }
        if (!passwordEncoder.matches(password, u.getPassword())) {
            throw new IllegalArgumentException("Invalid credentials");
        }

        // Простейшие клеймы — id + роли
        String roles = u.getRoles().stream().map(Role::getName).reduce((a,b)->a + "," + b).orElse("");

        return jwtService.generate(
                u.getEmail(),
                Map.of(
                        "uid", String.valueOf(u.getId()),
                        "name", u.getFullName() == null ? "" : u.getFullName(),
                        "roles", roles
                )
        );
    }
}
