package com.example.ailowcurrentengineer.controller;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
@RequestMapping("/api/jobs")
public class JobController {
    @PostMapping("/run")
    public ResponseEntity<?> run(@RequestBody Map<String, Object> req) {
// req: { projectId, preferencesJsonKey, rulesProfile }
// TODO: call planner /process pipeline (ingest->place->route->export)
        return ResponseEntity.accepted().body(Map.of("jobId", "JOB-123"));
    }

    @GetMapping("/{jobId}")
    public ResponseEntity<?> status(@PathVariable String jobId) {
// TODO: read job status
        return ResponseEntity.ok(Map.of("jobId", jobId, "status", "DONE",
                "exports", new String[]{"exports/drawings/project-1.pdf", "exports/drawings/project - 1.dxf" }));
    }
}