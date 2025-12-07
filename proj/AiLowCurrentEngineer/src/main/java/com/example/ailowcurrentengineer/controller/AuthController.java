// com/example/ailowcurrentengineer/controller/AuthController.java
package com.example.ailowcurrentengineer.controller;

import com.example.ailowcurrentengineer.service.UserService;
import jakarta.transaction.Transactional;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
@RequiredArgsConstructor
public class AuthController {

    private final UserService userService;

    @PostMapping("/api/public/register")
    @ResponseBody
    @Transactional
    public Map<String,Object> registerApi(@RequestBody Map<String,String> body) {
        String fullName = body.getOrDefault("fullName", "");
        String email = body.getOrDefault("email", "");
        String password = body.getOrDefault("password", "");
        userService.createUser(fullName, email, password);
        return Map.of("status","OK");
    }

    @PostMapping("/api/public/login")
    @ResponseBody
    public Map<String, Object> loginApi(@RequestBody Map<String, String> body) {
        String email = body.getOrDefault("email", "");
        String password = body.getOrDefault("password", "");
        String token = userService.login(email, password);
        return Map.of("token", token);
    }
}
