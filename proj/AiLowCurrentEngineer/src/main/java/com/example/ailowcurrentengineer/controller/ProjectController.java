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


    @PostMapping
    public ResponseEntity<CreateProjectResponse> create(@RequestBody String name) {
        Project p = projects.create(name);
        return ResponseEntity.ok(new CreateProjectResponse(p.getId().toString(), p.getName()));
    }

    @PostMapping("/{id}/upload")
    public ResponseEntity<UploadResponse> uploadByUrl(
            @PathVariable String id,
            @RequestBody Map<String, String> body) throws Exception {

        long pid = Long.parseLong(id);
        String pngUrl = body.get("pngUrl");

        byte[] data = webClient.get()
                .uri(pngUrl)
                .retrieve()
                .bodyToMono(byte[].class)
                .block();

        String key = id + "/" + UUID.randomUUID() + ".png";
        s3.uploadRawBytes(key, data, "image/png"); // <— этот метод должен класть в бакет raw-plans

        projects.setSrcKey(pid, key);
        planner.ingest(id, key);

        return ResponseEntity.ok(new UploadResponse(id, key));
    }
    @PostMapping("/{id}/infer")
    public PlannerDtos.InferResponse runInfer(
            @PathVariable String id,
            @RequestBody PlannerDtos.RunJobRequest req) {

        // projectId берём ИЗ PATH, а не из тела
        String projectId = id;

        // если в теле нет srcKey — возьми из БД (мы же его туда положили при upload)
        String srcKey = (req.srcKey() == null || req.srcKey().isBlank())
                ? projects.findById(Long.parseLong(id)).getSrcKey()
                : req.srcKey();

        // Принудительно прогоняем /ingest (мало ли гонки/старые данные)
        if (srcKey != null && !srcKey.isBlank()) {
            planner.ingest(projectId, srcKey);
        }

        // Собираем новый запрос с гарантированными полями
        PlannerDtos.RunJobRequest fixed = new PlannerDtos.RunJobRequest(
                projectId,
                srcKey,
                req.preferencesText(),
                req.exportFormats(),
                req.totalFixtures(),
                req.targetLux(),
                req.efficacyLmPerW(),
                req.maintenanceFactor(),
                req.utilizationFactor()
        );

        return planner.infer(fixed);
    }

}
