package com.example.ailowcurrentengineer.controller;

import com.example.ailowcurrentengineer.model.Project;
import com.example.ailowcurrentengineer.dto.PlannerDtos.CreateProjectResponse;
import com.example.ailowcurrentengineer.dto.PlannerDtos.UploadResponse;
import com.example.ailowcurrentengineer.service.PlannerClient;
import com.example.ailowcurrentengineer.service.ProjectService;
import com.example.ailowcurrentengineer.service.S3Service;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.UUID;

@RestController
@RequiredArgsConstructor
@RequestMapping("/api/projects")
public class ProjectController {

    private final S3Service s3;
    private final PlannerClient planner;
    private final ProjectService projects;


    @PostMapping
    public ResponseEntity<CreateProjectResponse> create(@RequestParam String name) {
        Project p = projects.create(name);
        return ResponseEntity.ok(new CreateProjectResponse(p.getId().toString(), p.getName()));
    }

    @PostMapping("/{id}/upload")
    public ResponseEntity<UploadResponse> upload(@PathVariable String id, @RequestParam MultipartFile file) throws Exception {
        Long pid = Long.getLong(id);

        // 1) кладём в MinIO
        String key = s3.uploadRawPlan(id, file);

        // 2) сохраняем srcKey в БД
        projects.setSrcKey(pid, key);

        // 3) дергаем Python /ingest
        planner.ingest(id, key);

        return ResponseEntity.ok(new UploadResponse(id, key));
    }
}
