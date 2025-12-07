package com.example.ailowcurrentengineer.controller;

import com.example.ailowcurrentengineer.model.Job;
import com.example.ailowcurrentengineer.dto.PlannerDtos.JobResponse;
import com.example.ailowcurrentengineer.dto.PlannerDtos.RunJobRequest;
import com.example.ailowcurrentengineer.service.JobService;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Map;
import java.util.UUID;

@RestController
@RequiredArgsConstructor
@RequestMapping("/api/jobs")
public class JobController {

    private final JobService jobService;
    private final ObjectMapper om = new ObjectMapper();

    @PostMapping("/run")
    public ResponseEntity<JobResponse> run(@RequestBody RunJobRequest req) {
        // создать Job(PENDING) → RUNNING → DONE/ERROR (синхронно)
        Job created = jobService.createPending(Long.valueOf(req.projectId()), req);
        Job finished = jobService.runNow(created, req);

        return ResponseEntity.accepted().body(mapJob(finished));
    }

    @GetMapping("/{jobId}")
    public ResponseEntity<JobResponse> status(@PathVariable String jobId) {
        Job j = jobService.get(Long.getLong(jobId));
        return ResponseEntity.ok(mapJob(j));
    }

    private JobResponse mapJob(Job j) {
        Map<String, Object> result = Map.of(
                "projectId", j.getProject().getId().toString(),
                "parsed", readJson(j.getParsedJson()),
                "lighting", readJson(j.getLightingJson()),
                "exported_files", readJson(j.getExportedFilesJson()),
                "uploaded_uris", readJson(j.getUploadedUrisJson()),
                "error", j.getErrorText()
        );
        return new JobResponse(j.getId().toString(), j.getStatus().name(), result);
    }

    private Object readJson(String s) {
        if (s == null) return null;
        try { return om.readValue(s, Object.class); }
        catch (Exception e) { return s; }
    }
}
