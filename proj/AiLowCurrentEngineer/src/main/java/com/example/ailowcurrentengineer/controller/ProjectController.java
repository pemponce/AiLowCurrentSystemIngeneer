package com.example.ailowcurrentengineer.controller;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.Map;

@RestController
@RequestMapping("/api/projects")
public class ProjectController {
    @PostMapping
    public ResponseEntity<?> create(@RequestParam String name) {
        // TODO: save Project to DB, return id
        return ResponseEntity.ok(Map.of("id",
                "00000000-0000-0000-0000-000000000001", "name", name));
    }

    @PostMapping("/{id}/upload")
    public ResponseEntity<?> upload(@PathVariable String id, @RequestParam
    MultipartFile file) {
        // TODO: put to MinIO raw-plans, return key
        return ResponseEntity.ok(Map.of("projectId", id, "key",
                file.getOriginalFilename()));
    }
}
