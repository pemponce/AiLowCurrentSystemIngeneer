package com.example.ailowcurrentengineer.service;

public interface UserService {
    void createUser(String fullName, String email, String password);
    String login(String email, String password); // <-- возвращаем JWT
}
