package com.example.ailowcurrentengineer.controller;

import com.example.ailowcurrentengineer.dto.PlannerDtos;
import com.example.ailowcurrentengineer.model.Project;
import com.example.ailowcurrentengineer.dto.PlannerDtos.CreateProjectResponse;
import com.example.ailowcurrentengineer.dto.PlannerDtos.UploadResponse;
import com.example.ailowcurrentengineer.service.PlannerClient;
import com.example.ailowcurrentengineer.service.ProjectService;
import com.example.ailowcurrentengineer.service.S3Service;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.reactive.function.client.WebClient;

import java.util.Map;
import java.util.UUID;

@RequiredArgsConstructor
@RestController
@RequestMapping("/api/projects")
public class ProjectController {
    private final S3Service s3;
    private final PlannerClient planner;
    private final ProjectService projects;
    private final WebClient webClient;

    @PostMapping("/{id}/upload")
    public ResponseEntity<UploadResponse> uploadByUrl(
            @PathVariable String id,
            @RequestBody Map<String, String> body) throws Exception {

        long pid = Long.parseLong(id); // ВАЖНО: не Long.getLong!

        String pngUrl = body.get("pngUrl");

        byte[] data = webClient.get()
                .uri(pngUrl)
                .retrieve()
                .bodyToMono(byte[].class)
                .block();

        String key = "raw-plans/" + id + "/" + UUID.randomUUID() + ".png";
        try (var in = new java.io.ByteArrayInputStream(data)) {
            // лучше складывать в RAW bucket, а не EXPORTS:
            s3.uploadRawBytes(key, data, "image/png");
        }

        projects.setSrcKey(pid, key);
        planner.ingest(id, key);

        return ResponseEntity.ok(new UploadResponse(id, key));
    }
    @PostMapping("/{id}/infer")
    public ResponseEntity<?> runInfer(@PathVariable String id, @RequestBody Map<String,Object> body){
        var req = new PlannerDtos.RunJobRequest(
                id,
                (String) body.getOrDefault("preferencesText", ""),
                (Integer) body.get("totalFixtures"),
                (Double) body.get("targetLux"),
                (Double) body.get("efficacyLmPerW"),
                (Double) body.get("maintenanceFactor"),
                (Double) body.get("utilizationFactor"),
                (java.util.List<String>) body.get("exportFormats")
        );
        var result = planner.infer(req);
        return ResponseEntity.ok(result);
    }

}
