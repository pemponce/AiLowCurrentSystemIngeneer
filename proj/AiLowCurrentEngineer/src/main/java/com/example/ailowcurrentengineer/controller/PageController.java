package com.example.ailowcurrentengineer.controller;

import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequiredArgsConstructor
public class PageController {

    @GetMapping("/login")
    public String login() {
        return "login"; // templates/login.html
    }

    @GetMapping("/register")
    public String register() {
        return "register"; // templates/register.html
    }

    @GetMapping("/")
    public String home() {
        return "redirect:/projects";
    }
}
